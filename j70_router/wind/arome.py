"""Météo-France AROME 0.01° wind subsets through GribStream."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil, floor
from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..sailing.geometry import LocalCartesian
from .base import WindSample
from .gridded import RegularGridWindField

MODEL = "aromefrance"
API_ROOT = "https://gribstream.com/api/v2"
U_SELECTOR = {"name": "U", "level": "10 m above ground", "info": "", "alias": "u_ms"}
V_SELECTOR = {"name": "V", "level": "10 m above ground", "info": "", "alias": "v_ms"}


class GribStreamConfigurationError(RuntimeError):
    pass


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("forecast datetimes must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class GribStreamClient:
    api_key: str | None = None
    session: requests.Session | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GRIBSTREAM_API_KEY")
        if not self.api_key:
            raise GribStreamConfigurationError(
                "GRIBSTREAM_API_KEY is not set; create a GribStream token and expose it as that environment variable"
            )
        self.session = self.session or requests.Session()

    def post_json(
        self, endpoint: str, payload: dict[str, Any], model: str = MODEL
    ) -> list[dict[str, Any]]:
        assert self.session is not None
        response = self.session.post(
            f"{API_ROOT}/{model}/{endpoint}",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError("GribStream returned a non-list JSON response")
        return rows


def get_latest_arome_run(
    client: GribStreamClient | None = None,
    now: datetime | None = None,
    probe_lat: float = 38.64,
    probe_lon: float = -9.50,
) -> datetime:
    """Return the newest AROME cycle actually available through GribStream."""

    client = client or GribStreamClient()
    now = _utc(now or datetime.now(timezone.utc))
    rows = client.post_json(
        "runs",
        {
            "forecastedFrom": _iso(now - timedelta(hours=18)),
            "forecastedUntil": _iso(now),
            "minLeadTime": "0h",
            "maxLeadTime": "1h",
            "coordinates": [{"lat": probe_lat, "lon": probe_lon}],
            "variables": [U_SELECTOR, V_SELECTOR],
        },
    )
    runs = {
        _parse_time(row["forecasted_at"])
        for row in rows
        if row.get("forecasted_at")
        and row.get("u_ms") is not None
        and row.get("v_ms") is not None
    }
    if not runs:
        raise RuntimeError(
            "AROME authenticated successfully, but no recent cycle has populated U/V "
            "at the course location. Cascais is outside the currently populated "
            "aromefrance footprint; use synthetic wind or another regional model."
        )
    return max(runs)


class AromeWindGrid:
    """Latitude/longitude API plus a local-frame adapter for the router."""

    def __init__(self, grid: RegularGridWindField, forecast_run: datetime, cache_path: Path) -> None:
        self.grid = grid
        self.forecast_run = forecast_run
        self.cache_path = cache_path
        self.model_name = "AROME 0.01°"
        self.model_code = MODEL
        self.grid_resolution_deg = 0.01

    def wind_at(self, lat: float, lon: float, utc_time: datetime) -> dict[str, float]:
        sample = self.grid.wind_at_latlon(lat, lon, utc_time)
        return {
            "u_ms": sample.u_mps,
            "v_ms": sample.v_mps,
            "speed_ms": sample.speed_mps,
            "speed_knots": sample.speed_mps / 0.514444,
            "direction_from_deg": sample.direction_deg,
        }

    def for_local_frame(self, frame: LocalCartesian) -> "LocalAromeWindField":
        return LocalAromeWindField(self, frame)


@dataclass(frozen=True)
class LocalAromeWindField:
    source: AromeWindGrid
    frame: LocalCartesian

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        lat, lon = self.frame.to_latlon(*position)
        values = self.source.wind_at(lat, lon, time)
        return WindSample(values["u_ms"], values["v_ms"])


def _rows_to_grid(rows: list[dict[str, Any]], run: datetime, cache_path: Path) -> AromeWindGrid:
    selected = [row for row in rows if _parse_time(row["forecasted_at"]) == run]
    if not selected:
        raise RuntimeError(f"response contained no rows for selected AROME cycle {_iso(run)}")
    times = sorted({_parse_time(row["forecasted_time"]) for row in selected})
    def coordinate(row: dict[str, Any], short: str, long: str) -> float:
        value = row.get(short, row.get(long))
        if value is None:
            raise RuntimeError(f"AROME response row is missing {short}/{long}")
        return float(value)

    lats = sorted({coordinate(row, "lat", "latitude") for row in selected})
    lons = sorted({coordinate(row, "lon", "longitude") for row in selected})
    u = np.full((len(times), len(lats), len(lons)), np.nan)
    v = np.full_like(u, np.nan)
    time_index = {value: index for index, value in enumerate(times)}
    lat_index = {value: index for index, value in enumerate(lats)}
    lon_index = {value: index for index, value in enumerate(lons)}
    for row in selected:
        index = (
            time_index[_parse_time(row["forecasted_time"])],
            lat_index[coordinate(row, "lat", "latitude")],
            lon_index[coordinate(row, "lon", "longitude")],
        )
        if row.get("u_ms") is None or row.get("v_ms") is None:
            continue
        u[index] = float(row["u_ms"])
        v[index] = float(row["v_ms"])
    if np.isnan(u).any() or np.isnan(v).any():
        missing = int(np.isnan(u).sum() + np.isnan(v).sum())
        raise RuntimeError(
            f"AROME response does not form a complete populated time/latitude/longitude grid "
            f"({missing} missing U/V values)"
        )
    frame = LocalCartesian(float(np.mean(lats)), float(np.mean(lons)))
    regular = RegularGridWindField(
        frame,
        np.asarray(lats),
        np.asarray(lons),
        tuple(times),
        u,
        v,
        metadata={"source": "GribStream", "model": MODEL, "forecast_run": _iso(run)},
    )
    return AromeWindGrid(regular, run, cache_path)


def download_wind_box(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_time: datetime,
    end_time: datetime,
    client: GribStreamClient | None = None,
    cache_dir: str | Path = "data/gribstream",
    forecast_run: datetime | None = None,
) -> AromeWindGrid:
    """Download/cache one 0.01° AROME cycle over a small bounding box."""

    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("minimum bounds must be less than maximum bounds")
    start_time, end_time = _utc(start_time), _utc(end_time)
    if start_time >= end_time:
        raise ValueError("start_time must be before end_time")
    client = client or GribStreamClient()
    run = _utc(forecast_run) if forecast_run else get_latest_arome_run(client)
    start_hour = start_time.replace(minute=0, second=0, microsecond=0)
    end_hour = end_time.replace(minute=0, second=0, microsecond=0)
    if end_hour < end_time:
        end_hour += timedelta(hours=1)
    min_lead = floor((start_hour - run).total_seconds() / 3600.0)
    max_lead = ceil((end_hour - run).total_seconds() / 3600.0)
    if min_lead < 0 or max_lead > 51:
        raise ValueError(
            f"requested window is outside the AROME +0…+51 h horizon for run {_iso(run)}"
        )

    request_descriptor = {
        "model": MODEL,
        "run": _iso(run),
        "bounds": [min_lat, max_lat, min_lon, max_lon],
        "start": _iso(start_hour),
        "end": _iso(end_hour),
        "step": 0.01,
        "variables": [U_SELECTOR, V_SELECTOR],
    }
    digest = hashlib.sha256(
        json.dumps(request_descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"aromefrance_{run:%Y%m%d%H}_{digest}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return _rows_to_grid(payload["rows"], run, cache_path)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
            # An interrupted ingest can expose a run before its values are
            # populated. Do not let that response poison future retries.
            cache_path.unlink()

    rows = client.post_json(
        "runs",
        {
            "timesList": [_iso(run)],
            "minLeadTime": f"{min_lead}h",
            "maxLeadTime": f"{max_lead}h",
            "grid": {
                "minLatitude": min_lat,
                "maxLatitude": max_lat,
                "minLongitude": min_lon,
                "maxLongitude": max_lon,
                "step": 0.01,
            },
            "variables": [U_SELECTOR, V_SELECTOR],
        },
    )
    result = _rows_to_grid(rows, run, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"request": request_descriptor, "rows": rows}, indent=2), encoding="utf-8"
    )
    return result


def load_cached_arome(cache_dir: str | Path = "data/gribstream") -> AromeWindGrid:
    """Load the newest complete cached AROME response without API access."""

    paths = sorted(Path(cache_dir).glob("aromefrance_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = _parse_time(payload["request"]["run"])
            return _rows_to_grid(payload["rows"], run, path)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not paths:
        raise FileNotFoundError(f"no cached AROME responses found under {Path(cache_dir)}")
    raise RuntimeError("no usable AROME cache: " + "; ".join(errors))
