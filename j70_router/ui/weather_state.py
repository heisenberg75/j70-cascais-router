"""Validation and state helpers for the local weather-selection UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable, MutableMapping, Protocol

import numpy as np


class WeatherSourceStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE_FOR_LOCATION = "NOT_AVAILABLE_FOR_LOCATION"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


class _Grid(Protocol):
    latitudes: np.ndarray
    longitudes: np.ndarray
    times: tuple[datetime, ...]
    u_mps: np.ndarray
    v_mps: np.ndarray


class WeatherDataset(Protocol):
    grid: _Grid


@dataclass(frozen=True)
class WeatherValidation:
    status: WeatherSourceStatus
    message: str

    @property
    def available(self) -> bool:
        return self.status is WeatherSourceStatus.AVAILABLE


def validate_weather_dataset(
    dataset: WeatherDataset | None,
    bbox: tuple[float, float, float, float],
    requested_time: datetime,
) -> WeatherValidation:
    """Check a loaded grid before it is made the active weather source.

    ``bbox`` is (min_lat, max_lat, min_lon, max_lon). Validation is deliberately
    independent of a particular provider so an unavailable high-resolution source
    cannot be mistaken for a usable forecast.
    """

    if dataset is None:
        return WeatherValidation(WeatherSourceStatus.DOWNLOAD_FAILED, "no forecast dataset was loaded")
    try:
        grid = dataset.grid
        lats = np.asarray(grid.latitudes, dtype=float)
        lons = np.asarray(grid.longitudes, dtype=float)
        times = tuple(grid.times)
        u = np.asarray(grid.u_mps, dtype=float)
        v = np.asarray(grid.v_mps, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return WeatherValidation(WeatherSourceStatus.DOWNLOAD_FAILED, "forecast dataset has an invalid grid")

    if not len(times) or not len(lats) or not len(lons) or not u.size or not v.size:
        return WeatherValidation(WeatherSourceStatus.DOWNLOAD_FAILED, "forecast grid is empty")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        return WeatherValidation(WeatherSourceStatus.NOT_AVAILABLE_FOR_LOCATION, "forecast contains null or non-finite U/V")

    min_lat, max_lat, min_lon, max_lon = bbox
    if min_lat < lats[0] or max_lat > lats[-1] or min_lon < lons[0] or max_lon > lons[-1]:
        return WeatherValidation(
            WeatherSourceStatus.NOT_AVAILABLE_FOR_LOCATION,
            "forecast grid does not cover the requested course area",
        )
    if requested_time < times[0] or requested_time > times[-1]:
        return WeatherValidation(
            WeatherSourceStatus.NOT_AVAILABLE_FOR_LOCATION,
            "forecast grid does not cover the requested valid time",
        )
    return WeatherValidation(WeatherSourceStatus.AVAILABLE, "forecast grid is valid")


def first_valid_weather_source(
    candidates: Iterable[tuple[str, WeatherDataset | None]],
    bbox: tuple[float, float, float, float],
    requested_time: datetime,
) -> tuple[str | None, WeatherDataset | None]:
    """Return the first usable loaded source in the caller's explicit priority order."""

    for name, candidate in candidates:
        if validate_weather_dataset(candidate, bbox, requested_time).available:
            return name, candidate
    return None, None


def change_forecast_time(state: MutableMapping[str, object], value: datetime) -> bool:
    """Change only the selected valid time and route state; retain loaded weather."""

    if state.get("selected_forecast_time") == value:
        return False
    state["selected_forecast_time"] = value
    state["last_route"] = None
    state["last_route_signature"] = None
    if "last_route_error" in state:
        state["last_route_error"] = None
    return True


def mark_update_invalidates_route(
    state: MutableMapping[str, object],
    prefix: str,
    lat: float,
    lon: float,
    tolerance_deg: float = 1e-7,
) -> bool:
    """Apply a drag update once; retain weather state and invalidate only route state."""

    lat_key, lon_key = f"{prefix}_lat", f"{prefix}_lon"
    if (
        abs(float(state[lat_key]) - lat) <= tolerance_deg
        and abs(float(state[lon_key]) - lon) <= tolerance_deg
    ):
        return False
    state[lat_key] = lat
    state[lon_key] = lon
    state["last_route"] = None
    state["last_route_signature"] = None
    if "last_route_error" in state:
        state["last_route_error"] = None
    return True
