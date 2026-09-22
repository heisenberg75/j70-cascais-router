"""Wind interfaces and meteorological direction conversions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan2, degrees, hypot, radians, sin, cos
from typing import Protocol

Position = tuple[float, float]


class WindOutOfBoundsError(ValueError):
    """Raised when a route state leaves the available forecast domain."""


@dataclass(frozen=True)
class WindSample:
    """A horizontal wind vector.

    ``u`` is positive east and ``v`` is positive north. ``direction_deg`` is
    meteorological direction: the compass bearing the wind COMES FROM.
    """

    u_mps: float
    v_mps: float

    @property
    def speed_mps(self) -> float:
        return hypot(self.u_mps, self.v_mps)

    @property
    def direction_deg(self) -> float:
        if self.speed_mps < 1e-12:
            return 0.0
        return degrees(atan2(-self.u_mps, -self.v_mps)) % 360.0

    @classmethod
    def from_speed_direction(cls, speed_mps: float, direction_deg: float) -> "WindSample":
        """Build from speed and meteorological (from) direction."""

        angle = radians(direction_deg)
        return cls(
            u_mps=-speed_mps * sin(angle),
            v_mps=-speed_mps * cos(angle),
        )


class WindField(Protocol):
    def wind_at(self, position: Position, time: datetime) -> WindSample:
        """Return wind at local Cartesian ``(east_m, north_m)`` and UTC time."""


@dataclass(frozen=True)
class TimeClampedWindField:
    """Hold the nearest forecast frame outside a field's time window."""

    source: WindField
    first_time: datetime
    last_time: datetime

    def __post_init__(self) -> None:
        if self.first_time > self.last_time:
            raise ValueError("first_time must not follow last_time")

    def wind_at(self, position: Position, time: datetime) -> WindSample:
        bounded_time = min(max(time, self.first_time), self.last_time)
        return self.source.wind_at(position, bounded_time)
