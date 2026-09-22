"""Regular latitude/longitude wind grids with U/V interpolation."""

from __future__ import annotations

import json
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..sailing.geometry import LocalCartesian
from .base import WindOutOfBoundsError, WindSample


class RegularGridWindField:
    def __init__(
        self,
        frame: LocalCartesian,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        times: tuple[datetime, ...],
        u_mps: np.ndarray,
        v_mps: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        self.frame = frame
        self.latitudes = np.asarray(latitudes, dtype=float)
        self.longitudes = np.asarray(longitudes, dtype=float)
        self.times = times
        self.u_mps = np.asarray(u_mps, dtype=float)
        self.v_mps = np.asarray(v_mps, dtype=float)
        self.metadata = metadata or {}
        expected = (len(times), len(self.latitudes), len(self.longitudes))
        if self.u_mps.shape != expected or self.v_mps.shape != expected:
            raise ValueError(f"wind arrays must have shape {expected}")
        if len(times) < 1 or len(self.latitudes) < 2 or len(self.longitudes) < 2:
            raise ValueError("grid needs at least one time and two points on each spatial axis")
        if not np.all(np.diff(self.latitudes) > 0) or not np.all(np.diff(self.longitudes) > 0):
            raise ValueError("latitude and longitude axes must be strictly increasing")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("forecast times must be strictly increasing")

    @classmethod
    def from_open_meteo_json(cls, path: str | Path, frame: LocalCartesian) -> "RegularGridWindField":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        speed = np.asarray(payload["wind_speed_mps"], dtype=float)
        direction = np.deg2rad(np.asarray(payload["wind_direction_deg"], dtype=float))
        # Meteorological direction is where wind comes from.
        u = -speed * np.sin(direction)
        v = -speed * np.cos(direction)
        times = tuple(
            datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            for value in payload["time_utc"]
        )
        metadata = {key: payload.get(key) for key in ("source", "model", "downloaded_at_utc", "request_url")}
        return cls(
            frame,
            np.asarray(payload["latitude"]),
            np.asarray(payload["longitude"]),
            times,
            u,
            v,
            metadata,
        )

    @staticmethod
    def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        if value < axis[0] or value > axis[-1]:
            raise WindOutOfBoundsError(
                f"query {value:.5f} is outside forecast grid [{axis[0]:.5f}, {axis[-1]:.5f}]"
            )
        upper = min(max(bisect_right(axis, value), 1), len(axis) - 1)
        lower = upper - 1
        weight = (value - axis[lower]) / (axis[upper] - axis[lower])
        return lower, upper, float(weight)

    def _time_bracket(self, time: datetime) -> tuple[int, int, float]:
        if time < self.times[0] or time > self.times[-1]:
            raise WindOutOfBoundsError(f"time {time.isoformat()} is outside forecast range")
        if len(self.times) == 1:
            return 0, 0, 0.0
        upper = min(max(bisect_right(self.times, time), 1), len(self.times) - 1)
        lower = upper - 1
        interval = (self.times[upper] - self.times[lower]).total_seconds()
        weight = (time - self.times[lower]).total_seconds() / interval
        return lower, upper, weight

    @staticmethod
    def _bilinear(array: np.ndarray, yi: int, yj: int, wy: float, xi: int, xj: int, wx: float) -> float:
        bottom = array[yi, xi] * (1.0 - wx) + array[yi, xj] * wx
        top = array[yj, xi] * (1.0 - wx) + array[yj, xj] * wx
        return float(bottom * (1.0 - wy) + top * wy)

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        lat, lon = self.frame.to_latlon(*position)
        return self.wind_at_latlon(lat, lon, time)

    def wind_at_latlon(self, lat: float, lon: float, time: datetime) -> WindSample:
        yi, yj, wy = self._bracket(self.latitudes, lat)
        xi, xj, wx = self._bracket(self.longitudes, lon)
        ti, tj, wt = self._time_bracket(time)
        u0 = self._bilinear(self.u_mps[ti], yi, yj, wy, xi, xj, wx)
        v0 = self._bilinear(self.v_mps[ti], yi, yj, wy, xi, xj, wx)
        if ti == tj:
            return WindSample(u0, v0)
        u1 = self._bilinear(self.u_mps[tj], yi, yj, wy, xi, xj, wx)
        v1 = self._bilinear(self.v_mps[tj], yi, yj, wy, xi, xj, wx)
        return WindSample(u0 * (1.0 - wt) + u1 * wt, v0 * (1.0 - wt) + v1 * wt)
