"""DWD ICON-EU approximately 7 km wind subsets through GribStream."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..sailing.geometry import LocalCartesian
from .arome import GribStreamClient
from .base import WindSample
from .gridded import RegularGridWindField

MODEL = "iconeu"
GRID_STEP = 0.0625
U_SELECTOR = {"name": "U_10M", "level": "10 m above ground", "info": "", "alias": "u_ms"}
V_SELECTOR = {"name": "V_10M", "level": "10 m above ground", "info": "", "alias": "v_ms"}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("forecast datetimes must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def get_latest_icon_eu_run(
    client: GribStreamClient | None = None,
    now: datetime | None = None,
    probe_lat: float = 38.64,
    probe_lon: float = -9.50,
) -> datetime:
    """Return the newest ICON-EU cycle with populated wind at Cascais."""

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
        model=MODEL,
    )
    runs = {
        _parse_time(row["forecasted_at"])
        for row in rows
        if row.get("forecasted_at")
        and row.get("u_ms") is not None
        and row.get("v_ms") is not None
    }
    if not runs:
        raise RuntimeError("GribStream returned no recent ICON-EU cycle with populated U/V at Cascais")
    return max(runs)


class IconEuWindGrid:
    def __init__(self, grid: RegularGridWindField, forecast_run: datetime, cache_path: Path) -> None:
        self.grid = grid
        self.forecast_run = forecast_run
        self.cache_path = cache_path
        self.model_name = "ICON-EU ~7 km"
        self.model_code = MODEL
        self.grid_resolution_deg = GRID_STEP

    def wind_at(self, lat: float, lon: float, utc_time: datetime) -> dict[str, float]:
        sample = self.grid.wind_at_latlon(lat, lon, utc_time)
        return {
            "u_ms": sample.u_mps,
            "v_ms": sample.v_mps,
            "speed_ms": sample.speed_mps,
            "speed_knots": sample.speed_mps / 0.514444,
            "direction_from_deg": sample.direction_deg,
        }

    def for_local_frame(self, frame: LocalCartesian) -> "LocalIconEuWindField":
        return LocalIconEuWindField(self, frame)


@dataclass(frozen=True)
class LocalIconEuWindField:
    source: IconEuWindGrid
    frame: LocalCartesian

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        lat, lon = self.frame.to_latlon(*position)
        value = self.source.wind_at(lat, lon, time)
        return WindSample(value["u_ms"], value["v_ms"])


def _rows_to_grid(rows: list[dict[str, Any]], run: datetime, cache_path: Path) -> IconEuWindGrid:
    selected = [row for row in rows if _parse_time(row["forecasted_at"]) == run]
    if not selected:
        raise RuntimeError(f"response contained no rows for selected ICON-EU cycle {_iso(run)}")
    times = sorted({_parse_time(row["forecasted_time"]) for row in selected})
    lats = sorted({float(row.get("lat", row.get("latitude"))) for row in selected})
    lons = sorted({float(row.get("lon", row.get("longitude"))) for row in selected})
    u = np.full((len(times), len(lats), len(lons)), np.nan)
    v = np.full_like(u, np.nan)
    ti = {value: index for index, value in enumerate(times)}
    yi = {value: index for index, value in enumerate(lats)}
    xi = {value: index for index, value in enumerate(lons)}
    for row in selected:
        if row.get("u_ms") is None or row.get("v_ms") is None:
            continue
        index = (
            ti[_parse_time(row["forecasted_time"])],
            yi[float(row.get("lat", row.get("latitude")))],
            xi[float(row.get("lon", row.get("longitude")))],
        )
        u[index], v[index] = float(row["u_ms"]), float(row["v_ms"])
    if np.isnan(u).any() or np.isnan(v).any():
        raise RuntimeError("ICON-EU response does not form a complete populated grid")
    grid = RegularGridWindField(
        LocalCartesian(float(np.mean(lats)), float(np.mean(lons))),
        np.asarray(lats), np.asarray(lons), tuple(times), u, v,
        metadata={"source": "GribStream", "model": MODEL, "forecast_run": _iso(run)},
    )
    return IconEuWindGrid(grid, run, cache_path)


def download_icon_eu_box(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_time: datetime,
    end_time: datetime,
    client: GribStreamClient | None = None,
    cache_dir: str | Path = "data/gribstream",
    forecast_run: datetime | None = None,
) -> IconEuWindGrid:
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("minimum bounds must be less than maximum bounds")
    start_time, end_time = _utc(start_time), _utc(end_time)
    client = client or GribStreamClient()
    run = _utc(forecast_run) if forecast_run else get_latest_icon_eu_run(client)
    min_lead = int((start_time - run).total_seconds() // 3600)
    max_lead = int(np.ceil((end_time - run).total_seconds() / 3600))
    if min_lead < 0 or max_lead > 120:
        raise ValueError("requested window is outside the ICON-EU +0…+120 h horizon")
    descriptor = {
        "model": MODEL, "run": _iso(run),
        "bounds": [min_lat, max_lat, min_lon, max_lon],
        "start": _iso(start_time), "end": _iso(end_time),
        "step": GRID_STEP, "variables": [U_SELECTOR, V_SELECTOR],
    }
    digest = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"iconeu_{run:%Y%m%d%H}_{digest}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return _rows_to_grid(payload["rows"], run, cache_path)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
            cache_path.unlink()
    rows = client.post_json(
        "runs",
        {
            "timesList": [_iso(run)],
            "minLeadTime": f"{min_lead}h",
            "maxLeadTime": f"{max_lead}h",
            "grid": {
                "minLatitude": min_lat, "maxLatitude": max_lat,
                "minLongitude": min_lon, "maxLongitude": max_lon,
                "step": GRID_STEP,
            },
            "variables": [U_SELECTOR, V_SELECTOR],
        },
        model=MODEL,
    )
    result = _rows_to_grid(rows, run, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"request": descriptor, "rows": rows}, indent=2), encoding="utf-8")
    return result


def load_cached_icon_eu(cache_dir: str | Path = "data/gribstream") -> IconEuWindGrid:
    paths = sorted(Path(cache_dir).glob("iconeu_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = _parse_time(payload["request"]["run"])
            return _rows_to_grid(payload["rows"], run, path)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
            continue
    raise FileNotFoundError(f"no usable cached ICON-EU responses found under {Path(cache_dir)}")
