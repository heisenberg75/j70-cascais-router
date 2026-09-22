"""Fast synthetic wind fields for routing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .base import Position, WindSample


@dataclass(frozen=True)
class SyntheticWindField:
    """A linear-in-space/time synthetic field.

    Spatial gradients use local Cartesian metres. Direction time rate is in
    degrees/hour, making Test B-D fields possible without weather downloads.
    """

    base_speed_mps: float = 8.0
    base_direction_deg: float = 0.0
    speed_gradient_x_per_m: float = 0.0
    speed_gradient_y_per_m: float = 0.0
    direction_gradient_x_deg_per_m: float = 0.0
    direction_gradient_y_deg_per_m: float = 0.0
    direction_rate_deg_per_hour: float = 0.0
    reference_time: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    minimum_speed_mps: float = 0.1

    @classmethod
    def uniform(cls, speed_mps: float, direction_deg: float) -> "SyntheticWindField":
        return cls(base_speed_mps=speed_mps, base_direction_deg=direction_deg)

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        x, y = position
        elapsed_hours = (time - self.reference_time).total_seconds() / 3600.0
        speed = max(
            self.minimum_speed_mps,
            self.base_speed_mps
            + self.speed_gradient_x_per_m * x
            + self.speed_gradient_y_per_m * y,
        )
        direction = (
            self.base_direction_deg
            + self.direction_gradient_x_deg_per_m * x
            + self.direction_gradient_y_deg_per_m * y
            + self.direction_rate_deg_per_hour * elapsed_hours
        ) % 360.0
        return WindSample.from_speed_direction(speed, direction)
