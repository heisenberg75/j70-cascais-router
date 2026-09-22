"""Theoretical and steering-uncertainty-aware target angles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from .j70_polar import J70Polar
from .polar import KNOT_TO_MPS

SailingMode = Literal["theoretical", "robust"]
LegMode = Literal["upwind", "downwind"]


def expected_vmg_knots(
    polar: J70Polar,
    tws_knots: float,
    commanded_twa_deg: float,
    mode: LegMode,
    heading_sigma_deg: float = 0.0,
    samples: int = 9,
) -> float:
    """Expected VMG under normally distributed heading error.

    Gauss-Hermite nodes integrate a normal distribution without random noise.
    The crew speed factor is intentionally absent: a uniform scale factor cannot
    change the angle that maximizes expected VMG.
    """

    if mode not in ("upwind", "downwind"):
        raise ValueError("mode must be 'upwind' or 'downwind'")
    if heading_sigma_deg < 0.0:
        raise ValueError("heading_sigma_deg cannot be negative")
    if samples < 3:
        raise ValueError("samples must be at least 3")
    if heading_sigma_deg == 0.0:
        angles = np.asarray([commanded_twa_deg])
        weights = np.asarray([1.0])
    else:
        nodes, raw_weights = np.polynomial.hermite.hermgauss(samples)
        angles = commanded_twa_deg + np.sqrt(2.0) * heading_sigma_deg * nodes
        angles = np.clip(angles, 0.0, polar.max_twa_deg)
        weights = raw_weights / np.sqrt(np.pi)
    speeds = np.asarray([polar.boat_speed(tws_knots, angle) for angle in angles])
    sign = 1.0 if mode == "upwind" else -1.0
    progress = sign * speeds * np.cos(np.deg2rad(angles))
    return float(np.sum(weights * progress))


@lru_cache(maxsize=4096)
def target_twa(
    polar: J70Polar,
    tws_knots: float,
    mode: LegMode,
    sailing_mode: SailingMode = "robust",
    heading_sigma_deg: float = 3.0,
) -> float:
    if sailing_mode not in ("theoretical", "robust"):
        raise ValueError("sailing_mode must be 'theoretical' or 'robust'")
    published_target = polar.published_target_twa(tws_knots, mode)
    if sailing_mode == "theoretical" or heading_sigma_deg == 0.0:
        return published_target
    sigma = heading_sigma_deg
    # ORC table speeds are rounded to 0.01 kt, while the published beat/run
    # targets come from the full-precision VPP. Anchor the uncertainty search
    # to that authoritative optimum so rounding cannot create another sailing
    # mode many degrees away. The robust command may move by up to one sigma.
    lower_limit, upper_limit = (25.0, 89.0) if mode == "upwind" else (91.0, polar.max_twa_deg)
    lower = max(lower_limit, published_target - heading_sigma_deg)
    upper = min(upper_limit, published_target + heading_sigma_deg)
    candidates = np.linspace(lower, upper, max(61, int((upper - lower) * 20) + 1))
    values = [expected_vmg_knots(polar, tws_knots, angle, mode, sigma) for angle in candidates]
    return float(candidates[int(np.argmax(values))])


@dataclass(eq=False)
class TargetAngleModel:
    polar: J70Polar
    sailing_mode: SailingMode = "robust"
    heading_sigma_deg: float = 3.0

    def __post_init__(self) -> None:
        if self.heading_sigma_deg < 0:
            raise ValueError("heading_sigma_deg cannot be negative")

    @lru_cache(maxsize=512)
    def _cached(self, tws_tenth_knot: int, mode: LegMode) -> float:
        return target_twa(
            self.polar,
            tws_tenth_knot / 10.0,
            mode,
            self.sailing_mode,
            self.heading_sigma_deg,
        )

    def __call__(self, tws_mps: float, mode: LegMode) -> float:
        return self._cached(round(tws_mps / KNOT_TO_MPS * 10.0), mode)
