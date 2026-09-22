"""Single-step trajectory propagation, independent of search strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol

from ..sailing.geometry import heading_vector, legal_heading, signed_twa
from ..wind.base import WindField, WindSample

SpeedModel = Callable[[float, float], float]
TargetModel = Callable[[float, str], float]


class CurrentField(Protocol):
    def current_at(self, position: tuple[float, float], time: datetime) -> tuple[float, float]: ...


@dataclass(frozen=True)
class TimeClampedCurrentField:
    """Hold the nearest available current frame outside its time window."""

    source: CurrentField
    first_time: datetime
    last_time: datetime

    def __post_init__(self) -> None:
        if self.first_time > self.last_time:
            raise ValueError("first_time must not follow last_time")

    def current_at(self, position: tuple[float, float], time: datetime) -> tuple[float, float]:
        bounded_time = min(max(time, self.first_time), self.last_time)
        return self.source.current_at(position, bounded_time)


@dataclass(frozen=True)
class StepResult:
    end: tuple[float, float]
    end_time: datetime
    heading_deg: float
    speed_mps: float
    wind: WindSample
    current_u_mps: float = 0.0
    current_v_mps: float = 0.0


def propagate(
    position: tuple[float, float],
    time: datetime,
    side: str,
    mode: str,
    dt_seconds: float,
    wind: WindField,
    speed_model: SpeedModel,
    target_model: TargetModel | None = None,
    current: CurrentField | None = None,
) -> StepResult:
    sample = wind.wind_at(position, time)
    if target_model is None:
        twa = signed_twa(mode, side)
        heading = legal_heading(sample.direction_deg, mode, side)
    else:
        target_twa = target_model(sample.speed_mps, mode)
        if not 0.0 < target_twa <= 180.0:
            raise ValueError("target angle model must return an angle in (0, 180]")
        twa = -target_twa if side == "port" else target_twa
        heading = (sample.direction_deg + twa) % 360.0
    speed = speed_model(sample.speed_mps, twa)
    if speed <= 0.0:
        raise ValueError("boat speed model must return a positive speed")
    vx, vy = heading_vector(heading)
    current_u, current_v = (0.0, 0.0) if current is None else current.current_at(position, time)
    end = (
        position[0] + (vx * speed + current_u) * dt_seconds,
        position[1] + (vy * speed + current_v) * dt_seconds,
    )
    return StepResult(
        end,
        time + timedelta(seconds=dt_seconds),
        heading,
        speed,
        sample,
        current_u,
        current_v,
    )
