from .geometry import LocalCartesian
from .j70_polar import J70Polar, J70SpeedModel
from .polar import ConstantSpeedPolar, WindScaledTestPolar, boat_speed
from .targets import TargetAngleModel, expected_vmg_knots, target_twa

__all__ = [
    "LocalCartesian",
    "ConstantSpeedPolar",
    "WindScaledTestPolar",
    "boat_speed",
    "J70Polar",
    "J70SpeedModel",
    "TargetAngleModel",
    "expected_vmg_knots",
    "target_twa",
]
