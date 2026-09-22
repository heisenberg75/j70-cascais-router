"""Copernicus Marine IBI hourly surface-current subsets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import atan2, degrees
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from ..sailing.geometry import LocalCartesian
from ..wind.base import WindOutOfBoundsError
from ..wind.gridded import RegularGridWindField

DATASET_ID = "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m"
GRID_RESOLUTION_DEG = 0.027
U_ALIASES = ("uo", "UO", "eastward_sea_water_velocity")
V_ALIASES = ("vo", "VO", "northward_sea_water_velocity")
UTC = timezone.utc


class CopernicusConfigurationError(RuntimeError):
    pass


def credentials_configured() -> bool:
    env = bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")) and bool(
        os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    )
    credential_dir = Path.home() / ".copernicusmarine"
    saved = any(
        (credential_dir / name).is_file()
        for name in (".copernicusmarine-credentials", ".netrc", "_netrc", "motuclient-python.ini")
    )
    return env or saved


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("current datetimes must be timezone-aware")
    return value.astimezone(UTC)


def _datetime(value: np.datetime64) -> datetime:
    seconds = np.datetime64(value, "s").astype("int64")
    return datetime.fromtimestamp(int(seconds), UTC)


def _coordinate_name(dataset: xr.Dataset, kind: str) -> str:
    aliases = {
        "time": ("time", "TIME"),
        "latitude": ("latitude", "lat", "LATITUDE"),
        "longitude": ("longitude", "lon", "LONGITUDE"),
    }[kind]
    for name in aliases:
        if name in dataset.coords:
            return name
    for name, coordinate in dataset.coords.items():
        if str(coordinate.attrs.get("standard_name", "")).lower() == kind:
            return name
    raise RuntimeError(f"Copernicus subset has no {kind} coordinate")


def _variable_name(dataset: xr.Dataset, aliases: tuple[str, ...], standard_name: str) -> str:
    for name in aliases:
        if name in dataset.data_vars:
            return name
    for name, variable in dataset.data_vars.items():
        if str(variable.attrs.get("standard_name", "")).lower() == standard_name:
            return name
    raise RuntimeError(f"Copernicus subset has no {standard_name} variable")


@dataclass(frozen=True)
class CopernicusCurrentGrid:
    grid: RegularGridWindField
    cache_path: Path
    downloaded_at: datetime
    model_name: str = "Copernicus Marine IBI · surface current"
    dataset_id: str = DATASET_ID
    grid_resolution_deg: float = GRID_RESOLUTION_DEG

    def _wet_spatial_value(
        self,
        field: np.ndarray,
        lat: float,
        lon: float,
    ) -> float:
        """Bilinear interpolation that ignores land-masked grid corners."""

        yi, yj, wy = self.grid._bracket(self.grid.latitudes, lat)
        xi, xj, wx = self.grid._bracket(self.grid.longitudes, lon)
        corners = (
            (yi, xi, (1.0 - wy) * (1.0 - wx)),
            (yi, xj, (1.0 - wy) * wx),
            (yj, xi, wy * (1.0 - wx)),
            (yj, xj, wy * wx),
        )
        populated = [
            (float(field[row, column]), weight)
            for row, column, weight in corners
            if np.isfinite(field[row, column]) and weight > 0.0
        ]
        if populated:
            total_weight = sum(weight for _, weight in populated)
            return sum(value * weight for value, weight in populated) / total_weight

        # Close to a complex shoreline all four enclosing cells can be masked.
        # Search only a small neighbourhood so a land position far from water
        # does not acquire a distant current value.
        centre_y = yi if abs(lat - self.grid.latitudes[yi]) <= abs(lat - self.grid.latitudes[yj]) else yj
        centre_x = xi if abs(lon - self.grid.longitudes[xi]) <= abs(lon - self.grid.longitudes[xj]) else xj
        for radius in range(1, 4):
            y0, y1 = max(0, centre_y - radius), min(len(self.grid.latitudes), centre_y + radius + 1)
            x0, x1 = max(0, centre_x - radius), min(len(self.grid.longitudes), centre_x + radius + 1)
            candidates: list[tuple[float, float]] = []
            for row in range(y0, y1):
                for column in range(x0, x1):
                    value = field[row, column]
                    if np.isfinite(value):
                        separation = (
                            (lat - self.grid.latitudes[row]) ** 2
                            + (lon - self.grid.longitudes[column]) ** 2
                        )
                        candidates.append((separation, float(value)))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        raise ValueError("current is unavailable at this land or missing-data grid cell")

    def _components_at(self, lat: float, lon: float, utc_time: datetime) -> tuple[float, float]:
        ti, tj, weight = self.grid._time_bracket(utc_time)

        def temporal(array: np.ndarray) -> float:
            first = self._wet_spatial_value(array[ti], lat, lon)
            if ti == tj:
                return first
            second = self._wet_spatial_value(array[tj], lat, lon)
            return first * (1.0 - weight) + second * weight

        return temporal(self.grid.u_mps), temporal(self.grid.v_mps)

    def current_at(self, lat: float, lon: float, utc_time: datetime) -> dict[str, float]:
        u_mps, v_mps = self._components_at(lat, lon, utc_time)
        speed = float(np.hypot(u_mps, v_mps))
        direction_to = 0.0 if speed < 1e-12 else degrees(atan2(u_mps, v_mps)) % 360.0
        return {
            "u_ms": u_mps,
            "v_ms": v_mps,
            "speed_ms": speed,
            "speed_knots": speed / 0.514444,
            "direction_to_deg": direction_to,
        }

    def for_local_frame(self, frame: LocalCartesian) -> "LocalCopernicusCurrentField":
        return LocalCopernicusCurrentField(self, frame)


@dataclass(frozen=True)
class LocalCopernicusCurrentField:
    source: CopernicusCurrentGrid
    frame: LocalCartesian

    def current_at(self, position: tuple[float, float], time: datetime) -> tuple[float, float]:
        lat, lon = self.frame.to_latlon(*position)
        try:
            value = self.source.current_at(lat, lon, time)
        except ValueError as exc:
            raise WindOutOfBoundsError(str(exc)) from exc
        return value["u_ms"], value["v_ms"]


def load_current_file(path: str | Path) -> CopernicusCurrentGrid:
    path = Path(path)
    with xr.open_dataset(path, engine="h5netcdf") as raw:
        dataset = raw.load()
    time_name = _coordinate_name(dataset, "time")
    lat_name = _coordinate_name(dataset, "latitude")
    lon_name = _coordinate_name(dataset, "longitude")
    u_name = _variable_name(dataset, U_ALIASES, "eastward_sea_water_velocity")
    v_name = _variable_name(dataset, V_ALIASES, "northward_sea_water_velocity")
    dataset = dataset.sortby([time_name, lat_name, lon_name])

    def values(name: str) -> np.ndarray:
        array = dataset[name]
        extras = [dimension for dimension in array.dims if dimension not in (time_name, lat_name, lon_name)]
        if extras:
            array = array.isel({dimension: 0 for dimension in extras})
        return np.asarray(array.transpose(time_name, lat_name, lon_name).values, dtype=float)

    latitudes = np.asarray(dataset[lat_name].values, dtype=float)
    longitudes = np.asarray(dataset[lon_name].values, dtype=float)
    times = tuple(_datetime(value) for value in np.asarray(dataset[time_name].values))
    if len(times) < 1 or len(latitudes) < 2 or len(longitudes) < 2:
        raise RuntimeError("Copernicus current subset is empty or too small")
    u, v = values(u_name), values(v_name)
    if not np.isfinite(u).any() or not np.isfinite(v).any():
        raise RuntimeError("Copernicus current subset contains no finite U/V values")
    downloaded = dataset.attrs.get("downloaded_at_utc")
    downloaded_at = datetime.fromisoformat(str(downloaded)) if downloaded else datetime.fromtimestamp(path.stat().st_mtime, UTC)
    grid = RegularGridWindField(
        LocalCartesian(float(np.mean(latitudes)), float(np.mean(longitudes))),
        latitudes,
        longitudes,
        times,
        u,
        v,
        metadata={"source": "Copernicus Marine", "dataset_id": DATASET_ID},
    )
    return CopernicusCurrentGrid(grid, path, downloaded_at)


def _open_remote(**kwargs) -> xr.Dataset:
    try:
        import copernicusmarine
    except ImportError as exc:
        raise CopernicusConfigurationError(
            "Copernicus Marine Toolbox is not installed; run pip install -r requirements.txt"
        ) from exc
    return copernicusmarine.open_dataset(**kwargs)


def download_current_box(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_time: datetime,
    end_time: datetime,
    cache_dir: str | Path = "data/copernicus",
    opener: Callable[..., xr.Dataset] | None = None,
) -> CopernicusCurrentGrid:
    """Download and cache the smallest IBI surface-current window covering the request."""

    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("minimum bounds must be less than maximum bounds")
    start_time, end_time = _utc(start_time), _utc(end_time)
    if end_time < start_time:
        raise ValueError("end_time must not precede start_time")
    if opener is None and not credentials_configured():
        raise CopernicusConfigurationError(
            "Copernicus credentials are not configured. Set COPERNICUSMARINE_SERVICE_USERNAME "
            "and COPERNICUSMARINE_SERVICE_PASSWORD or run 'copernicusmarine login'."
        )
    opener = opener or _open_remote
    remote = opener(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        coordinates_selection_method="outside",
    )
    time_name = _coordinate_name(remote, "time")
    available = tuple(_datetime(value) for value in np.asarray(remote[time_name].values))
    if not available:
        raise RuntimeError("Copernicus returned no current valid times")
    nearest = min(available, key=lambda value: abs((value - start_time).total_seconds()))
    if abs((nearest - start_time).total_seconds()) > 72 * 3600:
        raise RuntimeError(
            f"Copernicus IBI does not cover the requested time; nearest valid current is {nearest:%Y-%m-%d %H:%M UTC}"
        )
    before = max((value for value in available if value <= start_time), default=nearest)
    after_target = end_time + timedelta(hours=1)
    after = min((value for value in available if value >= after_target), default=available[-1])
    before64 = np.datetime64(before.astimezone(UTC).replace(tzinfo=None), "ns")
    after64 = np.datetime64(after.astimezone(UTC).replace(tzinfo=None), "ns")
    subset = remote.sel({time_name: slice(before64, after64)}).load()
    if subset.sizes.get(time_name, 0) < 1:
        raise RuntimeError("Copernicus returned an empty current time window")

    descriptor = {
        "dataset": DATASET_ID,
        "bounds": [round(min_lat, 5), round(max_lat, 5), round(min_lon, 5), round(max_lon, 5)],
        "first": before.isoformat(),
        "last": after.isoformat(),
    }
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ibi_current_{before:%Y%m%d%H}_{digest}.nc"
    subset.attrs["downloaded_at_utc"] = datetime.now(UTC).isoformat()
    subset.attrs["copernicus_dataset_id"] = DATASET_ID
    subset.to_netcdf(path, engine="h5netcdf")
    return load_current_file(path)


def load_cached_current(cache_dir: str | Path = "data/copernicus") -> CopernicusCurrentGrid:
    files = sorted(Path(cache_dir).glob("ibi_current_*.nc"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("no cached Copernicus Marine current subset was found")
    errors: list[str] = []
    for path in files:
        try:
            return load_current_file(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    raise RuntimeError("no valid cached Copernicus current subset: " + "; ".join(errors))
