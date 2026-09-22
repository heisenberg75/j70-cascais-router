"""Replaceable boat-speed models."""

from __future__ import annotations

from dataclasses import dataclass

KNOT_TO_MPS = 0.514444


def boat_speed(tws: float, twa: float, constant_speed_knots: float = 6.0) -> float:
    """Version-one speed model; arguments stay ready for a future J/70 polar."""

    del tws, twa
    return constant_speed_knots * KNOT_TO_MPS


@dataclass(frozen=True)
class ConstantSpeedPolar:
    speed_knots: float = 6.0

    def __call__(self, tws: float, twa: float) -> float:
        return boat_speed(tws, twa, self.speed_knots)


@dataclass(frozen=True)
class WindScaledTestPolar:
    """Simple synthetic-validation model, deliberately not a J/70 polar."""

    fraction_of_tws: float = 0.55
    minimum_speed_mps: float = 0.5
    maximum_speed_mps: float = 6.0

    def __call__(self, tws: float, twa: float) -> float:
        del twa
        return min(self.maximum_speed_mps, max(self.minimum_speed_mps, tws * self.fraction_of_tws))
