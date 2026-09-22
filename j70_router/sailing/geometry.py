"""Small-area geographic and sailing-angle geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin, sqrt, degrees

EARTH_RADIUS_M = 6_371_008.8
SIDES = ("port", "starboard")
MODES = ("upwind", "downwind")


@dataclass(frozen=True)
class LocalCartesian:
    """Equirectangular local frame, accurate for a compact race area."""

    origin_lat: float
    origin_lon: float

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        x = EARTH_RADIUS_M * radians(lon - self.origin_lon) * cos(radians(self.origin_lat))
        y = EARTH_RADIUS_M * radians(lat - self.origin_lat)
        return x, y

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        lat = self.origin_lat + degrees(y / EARTH_RADIUS_M)
        lon = self.origin_lon + degrees(x / (EARTH_RADIUS_M * cos(radians(self.origin_lat))))
        return lat, lon


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def heading_vector(heading_deg: float) -> tuple[float, float]:
    """Unit vector for navigation heading: 0 north, 90 east."""

    angle = radians(heading_deg)
    return sin(angle), cos(angle)


def signed_twa(mode: str, side: str) -> float:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    magnitude = 45.0 if mode == "upwind" else 135.0
    return -magnitude if side == "port" else magnitude


def legal_heading(wind_from_deg: float, mode: str, side: str) -> float:
    """Boat heading from meteorological true-wind direction and signed TWA."""

    return (wind_from_deg + signed_twa(mode, side)) % 360.0


def segment_circle_entry_fraction(
    start: tuple[float, float],
    end: tuple[float, float],
    centre: tuple[float, float],
    radius: float,
) -> float | None:
    """Earliest fraction along a segment that enters a circle, if any."""

    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    fx, fy = sx - centre[0], sy - centre[1]
    c = fx * fx + fy * fy - radius * radius
    if c <= 0.0:
        return 0.0
    a = dx * dx + dy * dy
    if a <= 1e-15:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = sqrt(discriminant)
    candidates = [t for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if 0.0 <= t <= 1.0]
    return min(candidates) if candidates else None
