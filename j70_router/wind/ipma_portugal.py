"""IPMA Portugal AROME-PT2 2.5 km operational 10 m wind provider."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import xarray as xr
from scipy.interpolate import griddata

from ..sailing.geometry import LocalCartesian
from .base import WindSample
from .gridded import RegularGridWindField

UTC = timezone.utc
BASE_URL = "https://mf2.ipma.pt/downloads/data/arome"
GRID_RESOLUTION_DEG = 0.025
MODEL_CODE = "AROME-PT2"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("forecast datetimes must be timezone-aware")
    return value.astimezone(UTC)


def run_urls(run: datetime) -> tuple[str, ...]:
    """Return the official PT2 10 m vector-wind products for a run.

    IPMA's public archive contains both ``UVCOMP`` and ``UV`` NetCDF files.
    ``UVCOMP`` is preferred because it explicitly denotes the two components,
    but either file is acceptable after metadata-driven U/V inspection.
    """

    run = _utc(run)
    stamp = run.strftime("%Y%m%d%H")
    directory = f"{BASE_URL}/{run:%Y/%m/%d}/run_{run:%H}"
    return (
        f"{directory}/AROME_OPER_001_FC_SP_PT2_025_UVCOMP_10-HTGL_{stamp}.nc",
        f"{directory}/AROME_OPER_001_FC_SP_PT2_025_UV_10-HTGL_{stamp}.nc",
    )


def run_url(run: datetime) -> str:
    """Return the preferred official PT2 component-file URL."""

    return run_urls(run)[0]


def _url_exists(url: str, timeout: float = 15.0) -> bool:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "j70-router/1.0"},
        stream=True,
        timeout=timeout,
    )
    try:
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"IPMA raw-data host denied access (HTTP {response.status_code}); "
                "the AROME-PT2 archive URL is valid but is not accessible from this network"
            )
        return response.status_code in (200, 206)
    finally:
        response.close()


def get_latest_ipma_portugal_run(
    now: datetime | None = None,
    exists: Callable[[str], bool] | None = None,
) -> datetime:
    """Return the newest actually published 00/06/12/18 UTC PT2 U/V run."""

    now = _utc(now or datetime.now(UTC))
    exists = exists or _url_exists
    first_hour = (now.hour // 6) * 6
    first = now.replace(hour=first_hour, minute=0, second=0, microsecond=0)
    for offset in range(16):
        candidate = first - timedelta(hours=6 * offset)
        for url in run_urls(candidate):
            if exists(url):
                return candidate
    raise RuntimeError("IPMA published no accessible AROME-PT2 10 m U/V run in the last four days")


def _coord_name(dataset: xr.Dataset, kind: str) -> str:
    aliases = {
        "time": ("time", "valid_time", "forecast_time"),
        "latitude": ("latitude", "lat", "nav_lat"),
        "longitude": ("longitude", "lon", "nav_lon"),
    }[kind]
    for name in aliases:
        if name in dataset.coords or name in dataset.variables:
            return name
    for name, variable in dataset.variables.items():
        standard = str(variable.attrs.get("standard_name", "")).lower()
        if standard == kind or (kind == "time" and standard == "time"):
            return name
    raise RuntimeError(f"IPMA NetCDF has no identifiable {kind} coordinate")


def _wind_variable(dataset: xr.Dataset, component: str) -> str:
    standard = "eastward_wind" if component == "u" else "northward_wind"
    aliases = (
        ("u10", "10u", "u", "u_wind", "x_wind_10m")
        if component == "u"
        else ("v10", "10v", "v", "v_wind", "y_wind_10m")
    )
    for name, variable in dataset.data_vars.items():
        attrs = " ".join(
            str(variable.attrs.get(key, ""))
            for key in ("standard_name", "long_name", "GRIB_shortName", "short_name")
        ).lower()
        normalized = name.lower()
        if str(variable.attrs.get("standard_name", "")).lower() == standard:
            return name
        if normalized in aliases:
            return name
        token = "eastward" if component == "u" else "northward"
        if token in attrs and "wind" in attrs:
            return name
        if component == "u" and re.search(r"(^|\W)(10u|u component)(\W|$)", attrs):
            return name
        if component == "v" and re.search(r"(^|\W)(10v|v component)(\W|$)", attrs):
            return name
    raise RuntimeError(f"IPMA NetCDF has no identifiable 10 m {component.upper()} wind component")


def _datetime(value: np.datetime64) -> datetime:
    seconds = np.datetime64(value, "s").astype("int64")
    return datetime.fromtimestamp(int(seconds), UTC)


def _forecast_layout(
    dataset: xr.Dataset,
    variable: xr.DataArray,
    horizontal_dims: tuple[str, str],
) -> tuple[str, tuple[datetime, ...], dict[str, int]]:
    """Identify the forecast dimension and its valid times.

    Operational NetCDF files commonly use either a sequence of ``time``
    values, or one initialization ``time`` plus a ``step`` dimension and a
    derived ``valid_time`` coordinate.  Treating initialization time as the
    forecast axis makes every forecast appear to be in the past.
    """

    extra_dims = [dim for dim in variable.dims if dim not in horizontal_dims]
    temporal: list[str] = []
    for dim in extra_dims:
        token = dim.lower()
        coordinate = dataset.coords.get(dim)
        is_temporal_dtype = coordinate is not None and (
            np.issubdtype(coordinate.dtype, np.datetime64)
            or np.issubdtype(coordinate.dtype, np.timedelta64)
        )
        if is_temporal_dtype or any(word in token for word in ("time", "step", "forecast")):
            temporal.append(dim)
    if not temporal:
        raise RuntimeError("IPMA wind variables have no identifiable forecast dimension")

    def priority(dim: str) -> tuple[int, int]:
        token = dim.lower()
        rank = 0 if "step" in token or "forecast_period" in token else 1
        return rank, -int(dataset.sizes[dim])

    time_dim = min(temporal, key=priority)
    selections = {
        dim: -1 if dim in temporal else 0
        for dim in extra_dims
        if dim != time_dim
    }
    count = int(dataset.sizes[time_dim])

    for name in ("valid_time", "forecast_time", time_dim, "time"):
        if name not in dataset.coords and name not in dataset.variables:
            continue
        coordinate = dataset[name]
        applicable = {dim: index for dim, index in selections.items() if dim in coordinate.dims}
        if applicable:
            coordinate = coordinate.isel(applicable)
        values = np.asarray(coordinate.values).squeeze()
        if values.ndim == 1 and values.size == count and np.issubdtype(values.dtype, np.datetime64):
            return time_dim, tuple(_datetime(value) for value in values), selections

    step = dataset[time_dim]
    step_values = np.asarray(step.values).squeeze()
    if np.issubdtype(step_values.dtype, np.timedelta64):
        for base_name in ("time", "forecast_reference_time"):
            if base_name in dataset.coords or base_name in dataset.variables:
                base = np.asarray(dataset[base_name].values).reshape(-1)[-1]
                if np.issubdtype(np.asarray(base).dtype, np.datetime64):
                    valid = np.asarray(base) + step_values
                    return time_dim, tuple(_datetime(value) for value in valid), selections
    raise RuntimeError("IPMA NetCDF has no usable forecast valid-time coordinate")


def _component_values(
    variable: xr.DataArray,
    time_name: str,
    horizontal_dims: tuple[str, str],
    selections: dict[str, int] | None = None,
) -> np.ndarray:
    if selections:
        variable = variable.isel({dim: value for dim, value in selections.items() if dim in variable.dims})
    extras = [dim for dim in variable.dims if dim not in (time_name, *horizontal_dims)]
    if extras:
        variable = variable.isel({dim: 0 for dim in extras})
    if time_name not in variable.dims:
        variable = variable.expand_dims({time_name: [0]})
    return np.asarray(variable.transpose(time_name, *horizontal_dims).values, dtype=float)


def load_ipma_portugal_file(
    path: str | Path,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    forecast_run: datetime | None = None,
) -> "IpmaPortugalWindGrid":
    """Inspect an IPMA NetCDF, identify U/V by metadata, and subset Cascais."""

    path = Path(path)
    with xr.open_dataset(path) as opened:
        dataset = opened.load()
    lat_name = _coord_name(dataset, "latitude")
    lon_name = _coord_name(dataset, "longitude")
    u_name, v_name = _wind_variable(dataset, "u"), _wind_variable(dataset, "v")
    lat = np.asarray(dataset[lat_name].values, dtype=float)
    lon = np.asarray(dataset[lon_name].values, dtype=float)
    if lat.ndim == lon.ndim == 1:
        y_dim, x_dim = dataset[lat_name].dims[0], dataset[lon_name].dims[0]
        time_name, times, selections = _forecast_layout(dataset, dataset[u_name], (y_dim, x_dim))
        y_idx = np.flatnonzero((lat >= min_lat) & (lat <= max_lat))
        x_idx = np.flatnonzero((lon >= min_lon) & (lon <= max_lon))
        if len(y_idx) < 2 or len(x_idx) < 2:
            raise RuntimeError("requested box is outside or too small for the IPMA PT2 grid")
        y_slice, x_slice = slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1)
        lats, lons = lat[y_slice], lon[x_slice]
        u = _component_values(dataset[u_name], time_name, (y_dim, x_dim), selections)[:, y_slice, x_slice]
        v = _component_values(dataset[v_name], time_name, (y_dim, x_dim), selections)[:, y_slice, x_slice]
        if lats[0] > lats[-1]:
            lats, u, v = lats[::-1], u[:, ::-1], v[:, ::-1]
        if lons[0] > lons[-1]:
            lons, u, v = lons[::-1], u[:, :, ::-1], v[:, :, ::-1]
    elif lat.ndim == lon.ndim == 2 and lat.shape == lon.shape:
        y_dim, x_dim = dataset[lat_name].dims[-2:]
        time_name, times, selections = _forecast_layout(dataset, dataset[u_name], (y_dim, x_dim))
        mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        rows, columns = np.where(mask)
        if len(rows) < 4:
            raise RuntimeError("requested box is outside or too small for the IPMA PT2 grid")
        ys = slice(max(0, rows.min() - 1), min(lat.shape[0], rows.max() + 2))
        xs = slice(max(0, columns.min() - 1), min(lat.shape[1], columns.max() + 2))
        source_lat, source_lon = lat[ys, xs], lon[ys, xs]
        raw_u = _component_values(dataset[u_name], time_name, (y_dim, x_dim), selections)[:, ys, xs]
        raw_v = _component_values(dataset[v_name], time_name, (y_dim, x_dim), selections)[:, ys, xs]
        lats = np.arange(min_lat, max_lat + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG)
        lons = np.arange(min_lon, max_lon + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG)
        target_lon, target_lat = np.meshgrid(lons, lats)
        points = np.column_stack((source_lon.ravel(), source_lat.ravel()))
        u = np.empty((raw_u.shape[0], len(lats), len(lons)))
        v = np.empty_like(u)
        for index in range(raw_u.shape[0]):
            for raw, output in ((raw_u[index], u[index]), (raw_v[index], v[index])):
                output[:] = griddata(points, raw.ravel(), (target_lon, target_lat), method="linear")
                missing = ~np.isfinite(output)
                if missing.any():
                    output[missing] = griddata(
                        points, raw.ravel(), (target_lon[missing], target_lat[missing]), method="nearest"
                    )
    else:
        raise RuntimeError("IPMA latitude/longitude coordinates have an unsupported shape")

    if len(times) < 1 or not np.isfinite(u).any() or not np.isfinite(v).any():
        raise RuntimeError("IPMA wind subset contains no populated U/V forecast data")
    run = _utc(forecast_run) if forecast_run else times[0]
    frame = LocalCartesian(float(np.mean(lats)), float(np.mean(lons)))
    grid = RegularGridWindField(
        frame, np.asarray(lats), np.asarray(lons), times, u, v,
        metadata={"source": "IPMA", "model": MODEL_CODE, "forecast_run": run.isoformat()},
    )
    return IpmaPortugalWindGrid(grid, run, path)


@dataclass(frozen=True)
class IpmaPortugalWindGrid:
    grid: RegularGridWindField
    forecast_run: datetime
    cache_path: Path
    model_name: str = "IPMA Portugal - AROME-PT2 2.5 km"
    model_code: str = MODEL_CODE
    grid_resolution_deg: float = GRID_RESOLUTION_DEG

    def wind_at(self, lat: float, lon: float, utc_time: datetime) -> dict[str, float]:
        sample = self.grid.wind_at_latlon(lat, lon, utc_time)
        return {
            "u_ms": sample.u_mps,
            "v_ms": sample.v_mps,
            "speed_ms": sample.speed_mps,
            "speed_knots": sample.speed_mps / 0.514444,
            "direction_from_deg": sample.direction_deg,
        }

    def for_local_frame(self, frame: LocalCartesian) -> "LocalIpmaPortugalWindField":
        return LocalIpmaPortugalWindField(self, frame)


@dataclass(frozen=True)
class LocalIpmaPortugalWindField:
    source: IpmaPortugalWindGrid
    frame: LocalCartesian

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        lat, lon = self.frame.to_latlon(*position)
        value = self.source.wind_at(lat, lon, time)
        return WindSample(value["u_ms"], value["v_ms"])


def _download(url: str, destination: Path, timeout: float = 180.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "j70-router/1.0"}) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary.replace(destination)


def download_ipma_portugal_box(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_time: datetime,
    end_time: datetime,
    cache_dir: str | Path = "data/ipma",
    forecast_run: datetime | None = None,
    downloader: Callable[[str, Path], None] | None = None,
) -> IpmaPortugalWindGrid:
    del start_time, end_time  # the official file contains the complete hourly run
    run = _utc(forecast_run) if forecast_run else get_latest_ipma_portugal_run()
    directory = Path(cache_dir)
    full_path = directory / f"ipma_pt2_{run:%Y%m%d%H}_uv10.nc"
    if not full_path.exists():
        download = downloader or _download
        failures: list[str] = []
        for url in run_urls(run):
            try:
                download(url, full_path)
                break
            except Exception as exc:
                failures.append(f"{url}: {exc}")
        else:
            raise RuntimeError(
                "IPMA AROME-PT2 run exists but neither official 10 m U/V "
                "NetCDF product could be downloaded: " + "; ".join(failures)
            )
    descriptor = [run.isoformat(), min_lat, max_lat, min_lon, max_lon]
    digest = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()[:12]
    subset_path = directory / f"ipma_pt2_{run:%Y%m%d%H}_{digest}.npz"
    if subset_path.exists():
        with np.load(subset_path) as cached:
            times = tuple(datetime.fromtimestamp(float(value), UTC) for value in cached["times"])
            grid = RegularGridWindField(
                LocalCartesian(float(cached["frame_lat"]), float(cached["frame_lon"])),
                cached["latitudes"], cached["longitudes"], times, cached["u"], cached["v"],
                metadata={"source": "IPMA", "model": MODEL_CODE, "forecast_run": run.isoformat()},
            )
            return IpmaPortugalWindGrid(grid, run, subset_path)
    result = load_ipma_portugal_file(full_path, min_lat, max_lat, min_lon, max_lon, run)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subset_path,
        latitudes=result.grid.latitudes,
        longitudes=result.grid.longitudes,
        times=np.asarray([value.timestamp() for value in result.grid.times]),
        u=result.grid.u_mps,
        v=result.grid.v_mps,
        frame_lat=result.grid.frame.origin_lat,
        frame_lon=result.grid.frame.origin_lon,
    )
    return IpmaPortugalWindGrid(result.grid, run, subset_path)


def load_cached_ipma_portugal(cache_dir: str | Path = "data/ipma") -> IpmaPortugalWindGrid:
    directory = Path(cache_dir)
    subsets = sorted(directory.glob("ipma_pt2_*_*.npz"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not subsets:
        raise FileNotFoundError("no cached IPMA Portugal AROME-PT2 subset was found")
    path = subsets[0]
    match = re.search(r"ipma_pt2_(\d{10})_", path.name)
    if not match:
        raise RuntimeError(f"invalid IPMA cache filename: {path.name}")
    run = datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=UTC)
    with np.load(path) as cached:
        times = tuple(datetime.fromtimestamp(float(value), UTC) for value in cached["times"])
        grid = RegularGridWindField(
            LocalCartesian(float(cached["frame_lat"]), float(cached["frame_lon"])),
            cached["latitudes"], cached["longitudes"], times, cached["u"], cached["v"],
            metadata={"source": "IPMA", "model": MODEL_CODE, "forecast_run": run.isoformat()},
        )
    return IpmaPortugalWindGrid(grid, run, path)
