"""Interpolated J/70 ORC best-performance polar."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .polar import KNOT_TO_MPS

DEFAULT_POLAR_PATH = Path(__file__).parents[2] / "data" / "polar" / "j70_orc_best_performance.json"


class J70Polar:
    """Two-dimensional polar with bounded interpolation.

    TWS is clamped to the published 6--20 kt range. TWA is mirrored to
    0--180 degrees. Within each TWS row, linear interpolation avoids overshoot
    between published values. Beyond the deepest published angle, the final
    downward slope is continued and floored at zero; holding the final speed
    constant would create a false dead-downwind VMG maximum. The first point five degrees
    below the published beat target is assigned zero speed to represent the
    no-go zone rather than inventing close-hauled performance.
    """

    def __init__(self, path: str | Path = DEFAULT_POLAR_PATH) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.metadata = {key: value for key, value in raw.items() if key != "rows"}
        self.tws_values = np.asarray([row["tws"] for row in raw["rows"]], dtype=float)
        self.beat_twa_values = np.asarray([row["beat_twa"] for row in raw["rows"]], dtype=float)
        self.run_twa_values = np.asarray([row["run_twa"] for row in raw["rows"]], dtype=float)
        common = np.asarray(raw["common_twa"], dtype=float)
        self.max_twa_deg = 180.0
        self._curves: list[tuple[np.ndarray, np.ndarray]] = []
        for row in raw["rows"]:
            angles = [max(0.0, row["beat_twa"] - 5.0), row["beat_twa"], *common]
            speeds = [0.0, row["beat_speed"], *row["speeds"]]
            angles.append(row["run_twa"])
            speeds.append(row["run_speed"])
            ordered = sorted(zip(angles, speeds), key=lambda item: item[0])
            # The run target can coincide with a standard angle; retain its
            # published target value at that angle.
            merged: dict[float, float] = {}
            for angle, speed in ordered:
                merged[float(angle)] = float(speed)
            self._curves.append(
                (np.asarray(list(merged), dtype=float), np.asarray(list(merged.values()), dtype=float))
            )

    def boat_speed(self, tws_knots: float, twa_deg: float) -> float:
        tws = float(np.clip(tws_knots, self.tws_values[0], self.tws_values[-1]))
        twa = abs(float(twa_deg)) % 360.0
        if twa > 180.0:
            twa = 360.0 - twa
        row_values: list[float] = []
        for angles, speeds in self._curves:
            if twa <= angles[-1]:
                value = np.interp(twa, angles, speeds, left=0.0)
            else:
                slope = (speeds[-1] - speeds[-2]) / (angles[-1] - angles[-2])
                value = speeds[-1] + slope * (twa - angles[-1])
            row_values.append(max(0.0, float(value)))
        row_speeds = np.asarray(row_values)
        return float(np.interp(tws, self.tws_values, row_speeds))

    def __call__(self, tws_knots: float, twa_deg: float) -> float:
        return self.boat_speed(tws_knots, twa_deg)

    def published_target_twa(self, tws_knots: float, mode: str) -> float:
        """Interpolate the official ORC beat or gybe target angle."""

        if mode not in ("upwind", "downwind"):
            raise ValueError("mode must be 'upwind' or 'downwind'")
        values = self.beat_twa_values if mode == "upwind" else self.run_twa_values
        tws = float(np.clip(tws_knots, self.tws_values[0], self.tws_values[-1]))
        return float(np.interp(tws, self.tws_values, values))


@dataclass(frozen=True)
class J70SpeedModel:
    """Router adapter: SI inputs/output, with a separate crew speed factor."""

    polar: J70Polar
    crew_speed_factor: float = 0.93

    def __post_init__(self) -> None:
        if not 0.0 < self.crew_speed_factor <= 1.5:
            raise ValueError("crew_speed_factor must be positive and plausible")

    def __call__(self, tws_mps: float, twa_deg: float) -> float:
        speed_knots = self.polar.boat_speed(tws_mps / KNOT_TO_MPS, abs(twa_deg))
        return speed_knots * self.crew_speed_factor * KNOT_TO_MPS
