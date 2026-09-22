"""Maneuver-constrained numerical routing for one leg or one lap."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import atan2, ceil, degrees
from time import perf_counter
from typing import Literal

from ..sailing.geometry import SIDES, distance, segment_circle_entry_fraction
from ..wind.base import WindField, WindOutOfBoundsError, WindSample
from .simulator import CurrentField, SpeedModel, TargetModel, propagate

ManeuverConstraint = Literal["maximum", "exact"]


@dataclass(frozen=True)
class RoutingConfig:
    dt_seconds: float = 10.0
    spatial_bin_m: float = 30.0
    mark_radius_m: float = 15.0
    beam_width: int = 6_000
    max_leg_time_seconds: float = 7_200.0
    heuristic_speed_mps: float = 10.0
    maneuver_penalty_seconds: float = 0.0
    maneuver_constraint: ManeuverConstraint = "maximum"
    minimum_maneuvers: int = 0
    minimum_maneuver_separation_seconds: float = 0.0
    maneuver_preference_tolerance_seconds: float = 1.0
    max_computation_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.dt_seconds <= 0 or self.spatial_bin_m <= 0 or self.mark_radius_m <= 0:
            raise ValueError("dt_seconds, spatial_bin_m, and mark_radius_m must be positive")
        if self.beam_width < 1 or self.max_leg_time_seconds <= 0 or self.heuristic_speed_mps <= 0:
            raise ValueError("beam_width, max_leg_time_seconds, and heuristic_speed_mps must be positive")
        if self.maneuver_penalty_seconds < 0:
            raise ValueError("maneuver penalty cannot be negative")
        if self.minimum_maneuvers < 0:
            raise ValueError("minimum_maneuvers cannot be negative")
        if self.minimum_maneuver_separation_seconds < 0:
            raise ValueError("minimum maneuver separation cannot be negative")
        if self.maneuver_preference_tolerance_seconds < 0:
            raise ValueError("maneuver preference tolerance cannot be negative")
        if self.max_computation_seconds is not None and self.max_computation_seconds <= 0:
            raise ValueError("max computation seconds must be positive when set")
        if self.maneuver_constraint not in ("maximum", "exact"):
            raise ValueError("maneuver_constraint must be 'maximum' or 'exact'")


def adaptive_routing_config(config: RoutingConfig, leg_distance_m: float) -> RoutingConfig:
    """Scale search resolution so multi-kilometre legs remain interactive.

    Segment/circle intersection still determines the precise mark arrival, so
    a longer propagation step does not require landing on the mark. User
    settings remain lower bounds on fidelity.
    """

    if leg_distance_m < 0:
        raise ValueError("leg_distance_m cannot be negative")
    # Aim for roughly 220 generations at a conservative 1.5 m/s VMG.
    desired_dt = ceil(leg_distance_m / (1.5 * 220.0) / 5.0) * 5.0
    dt_seconds = max(config.dt_seconds, min(30.0, desired_dt))
    spatial_bin_m = max(config.spatial_bin_m, dt_seconds * 3.0)
    beam_cap = 1_200 if leg_distance_m < 3_000.0 else 400
    max_leg_time_seconds = max(
        config.max_leg_time_seconds,
        min(14_400.0, leg_distance_m / 0.8 + 1_800.0),
    )
    computation_budget = min(30.0, 10.0 + leg_distance_m / 1_000.0)
    if config.max_computation_seconds is None:
        max_computation_seconds = None
    else:
        max_computation_seconds = max(config.max_computation_seconds, computation_budget)
    return replace(
        config,
        dt_seconds=dt_seconds,
        spatial_bin_m=spatial_bin_m,
        beam_width=min(config.beam_width, beam_cap),
        max_leg_time_seconds=max_leg_time_seconds,
        max_computation_seconds=max_computation_seconds,
    )


@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    time: datetime
    side: str
    heading_deg: float | None = None
    boat_speed_mps: float | None = None
    wind: WindSample | None = None
    current_u_mps: float = 0.0
    current_v_mps: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class Maneuver:
    x: float
    y: float
    time: datetime
    from_side: str
    to_side: str


@dataclass(frozen=True)
class RouteResult:
    mode: str
    points: tuple[RoutePoint, ...]
    elapsed_seconds: float
    distance_m: float
    maneuvers_used: int
    states_evaluated: int = 0

    @property
    def arrival_time(self) -> datetime:
        return self.points[-1].time

    @property
    def maneuvers(self) -> tuple[Maneuver, ...]:
        found: list[Maneuver] = []
        for previous, current in zip(self.points, self.points[1:]):
            if previous.side != current.side:
                found.append(
                    Maneuver(previous.x, previous.y, previous.time, previous.side, current.side)
                )
        return tuple(found)


@dataclass(frozen=True)
class RaceResult:
    upwind: RouteResult
    downwind: RouteResult
    upwind_baselines: tuple["TacticalOption", ...] = ()
    downwind_baselines: tuple["TacticalOption", ...] = ()

    @property
    def total_elapsed_seconds(self) -> float:
        return self.upwind.elapsed_seconds + self.downwind.elapsed_seconds

    @property
    def total_distance_m(self) -> float:
        return self.upwind.distance_m + self.downwind.distance_m

    @property
    def states_evaluated(self) -> int:
        return self.upwind.states_evaluated + self.downwind.states_evaluated


@dataclass(frozen=True)
class TacticalOption:
    initial_side: str
    course_side: str
    route: RouteResult


@dataclass(frozen=True)
class MultiLapResult:
    legs: tuple[RouteResult, ...]
    laps: int
    lap_results: tuple[RaceResult, ...] = ()

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(leg.elapsed_seconds for leg in self.legs)

    @property
    def total_distance_m(self) -> float:
        return sum(leg.distance_m for leg in self.legs)

    @property
    def arrival_time(self) -> datetime:
        return self.legs[-1].arrival_time

    @property
    def tacks(self) -> int:
        return sum(leg.maneuvers_used for leg in self.legs if leg.mode == "upwind")

    @property
    def gybes(self) -> int:
        return sum(leg.maneuvers_used for leg in self.legs if leg.mode == "downwind")

    @property
    def states_evaluated(self) -> int:
        return sum(leg.states_evaluated for leg in self.legs)


@dataclass(frozen=True)
class _Node:
    x: float
    y: float
    time: datetime
    elapsed_seconds: float
    side: str
    maneuvers: int
    distance_m: float
    moves_on_side: int
    parent: "_Node | None"
    heading_deg: float | None = None
    boat_speed_mps: float | None = None
    wind: WindSample | None = None
    current_u_mps: float = 0.0
    current_v_mps: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


class RouteNotFoundError(RuntimeError):
    pass


def heading_sector_margin_deg(
    start: tuple[float, float],
    target: tuple[float, float],
    wind_from_deg: float,
    target_twa_deg: float,
    mode: Literal["upwind", "downwind"],
) -> float:
    """Return how far the course bearing lies inside the locally sailable sector.

    A positive result is inside the cone spanned by the two legal headings and
    a negative result is outside it. This is a UI diagnostic rather than a
    replacement for trajectory search: varying wind can alter reachability.
    """

    if mode not in ("upwind", "downwind"):
        raise ValueError("mode must be 'upwind' or 'downwind'")
    if not 0.0 < target_twa_deg < 180.0:
        raise ValueError("target_twa_deg must be between 0 and 180 degrees")
    dx, dy = target[0] - start[0], target[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        return 180.0
    course_bearing = degrees(atan2(dx, dy)) % 360.0
    sector_center = wind_from_deg % 360.0 if mode == "upwind" else (wind_from_deg + 180.0) % 360.0
    sector_half_width = target_twa_deg if mode == "upwind" else 180.0 - target_twa_deg
    angular_offset = abs((course_bearing - sector_center + 180.0) % 360.0 - 180.0)
    return sector_half_width - angular_offset


def _course_side(
    route: RouteResult,
    start: tuple[float, float],
    target: tuple[float, float],
) -> str:
    dx, dy = target[0] - start[0], target[1] - start[1]
    signed_offsets = [
        dx * (point.y - start[1]) - dy * (point.x - start[0])
        for point in route.points
    ]
    largest = max(signed_offsets, key=abs, default=0.0)
    return "left" if largest > 0.0 else "right"


def one_maneuver_options(
    start: tuple[float, float],
    target: tuple[float, float],
    start_time: datetime,
    wind: WindField,
    speed_model: SpeedModel,
    mode: Literal["upwind", "downwind"],
    config: RoutingConfig,
    target_model: TargetModel | None = None,
    current: CurrentField | None = None,
) -> tuple[TacticalOption, ...]:
    """Solve port-first and starboard-first one-maneuver tactical baselines."""

    baseline_config = replace(
        config,
        beam_width=min(config.beam_width, 160),
        maneuver_constraint="exact",
        minimum_maneuvers=0,
    )
    options: list[TacticalOption] = []
    for initial_side in SIDES:
        try:
            route = route_leg(
                start,
                target,
                start_time,
                wind,
                speed_model,
                mode=mode,
                max_maneuvers=1,
                config=baseline_config,
                target_model=target_model,
                current=current,
                initial_sides=(initial_side,),
            )
        except RouteNotFoundError:
            continue
        options.append(TacticalOption(initial_side, _course_side(route, start, target), route))
    return tuple(options)


def _node_key(node: _Node, bin_size: float) -> tuple[int, int, str, int]:
    return (
        round(node.x / bin_size),
        round(node.y / bin_size),
        node.side,
        node.maneuvers,
    )


def _reconstruct(node: _Node, mode: str, states_evaluated: int) -> RouteResult:
    nodes: list[_Node] = []
    cursor: _Node | None = node
    while cursor is not None:
        nodes.append(cursor)
        cursor = cursor.parent
    nodes.reverse()
    points = tuple(
        RoutePoint(
            n.x, n.y, n.time, n.side, n.heading_deg, n.boat_speed_mps, n.wind,
            n.current_u_mps, n.current_v_mps,
        )
        for n in nodes
    )
    return RouteResult(mode, points, node.elapsed_seconds, node.distance_m, node.maneuvers, states_evaluated)


def route_leg(
    start: tuple[float, float],
    target: tuple[float, float],
    start_time: datetime,
    wind: WindField,
    speed_model: SpeedModel,
    mode: Literal["upwind", "downwind"] = "upwind",
    max_maneuvers: int = 4,
    config: RoutingConfig | None = None,
    target_model: TargetModel | None = None,
    current: CurrentField | None = None,
    initial_sides: tuple[str, ...] = SIDES,
) -> RouteResult:
    """Find the fastest discrete trajectory subject to a maneuver constraint."""

    if max_maneuvers < 0:
        raise ValueError("max_maneuvers cannot be negative")
    if mode not in ("upwind", "downwind"):
        raise ValueError("mode must be 'upwind' or 'downwind'")
    if not initial_sides or any(side not in SIDES for side in initial_sides):
        raise ValueError("initial_sides must contain port and/or starboard")
    cfg = config or RoutingConfig()
    if cfg.minimum_maneuvers > max_maneuvers:
        raise ValueError("minimum_maneuvers cannot exceed max_maneuvers")
    if distance(start, target) <= cfg.mark_radius_m:
        if (cfg.maneuver_constraint == "exact" and max_maneuvers != 0) or (
            cfg.maneuver_constraint == "maximum" and cfg.minimum_maneuvers > 0
        ):
            raise RouteNotFoundError("already at mark before exact maneuver count can be met")
        point = RoutePoint(start[0], start[1], start_time, "port")
        return RouteResult(mode, (point,), 0.0, 0.0, 0)

    frontier = [
        _Node(start[0], start[1], start_time, 0.0, side, 0, 0.0, 0, None)
        for side in initial_sides
    ]
    max_steps = ceil(cfg.max_leg_time_seconds / cfg.dt_seconds)
    best_arrival: _Node | None = None
    states_evaluated = 0
    deadline = (
        perf_counter() + cfg.max_computation_seconds
        if cfg.max_computation_seconds is not None
        else None
    )

    for _ in range(max_steps):
        if deadline is not None and perf_counter() >= deadline:
            raise RouteNotFoundError(
                f"{mode} route search stopped after {cfg.max_computation_seconds:.1f} seconds; "
                "the mark may be unreachable on the allowed headings or the search settings are too fine"
            )
        deduplicated: dict[tuple[int, int, str, int], _Node] = {}
        for state in frontier:
            if deadline is not None and states_evaluated % 256 == 0 and perf_counter() >= deadline:
                raise RouteNotFoundError(
                    f"{mode} route search stopped after {cfg.max_computation_seconds:.1f} seconds; "
                    "the mark may be unreachable on the allowed headings or the search settings are too fine"
                )
            choices = [state.side]
            separation_met = (
                state.moves_on_side * cfg.dt_seconds >= cfg.minimum_maneuver_separation_seconds
            )
            if state.moves_on_side > 0 and separation_met and state.maneuvers < max_maneuvers:
                choices.append("starboard" if state.side == "port" else "port")

            for side in choices:
                states_evaluated += 1
                switched = side != state.side
                maneuvers = state.maneuvers + int(switched)
                penalty = cfg.maneuver_penalty_seconds if switched else 0.0
                move_time = state.time + timedelta(seconds=penalty)
                try:
                    step = propagate(
                        state.position,
                        move_time,
                        side,
                        mode,
                        cfg.dt_seconds,
                        wind,
                        speed_model,
                        target_model,
                        current,
                    )
                except WindOutOfBoundsError:
                    continue
                entry = segment_circle_entry_fraction(
                    state.position, step.end, target, cfg.mark_radius_m
                )
                segment_length = distance(state.position, step.end)
                if entry is not None:
                    allowed = (
                        (cfg.maneuver_constraint == "maximum" and maneuvers >= cfg.minimum_maneuvers)
                        or (cfg.maneuver_constraint == "exact" and maneuvers == max_maneuvers)
                    )
                    if allowed:
                        arrival = _Node(
                            x=state.x + (step.end[0] - state.x) * entry,
                            y=state.y + (step.end[1] - state.y) * entry,
                            time=move_time + timedelta(seconds=cfg.dt_seconds * entry),
                            elapsed_seconds=state.elapsed_seconds + penalty + cfg.dt_seconds * entry,
                            side=side,
                            maneuvers=maneuvers,
                            distance_m=state.distance_m + segment_length * entry,
                            moves_on_side=(0 if switched else state.moves_on_side) + 1,
                            parent=state,
                            heading_deg=step.heading_deg,
                            boat_speed_mps=step.speed_mps,
                            wind=step.wind,
                            current_u_mps=step.current_u_mps,
                            current_v_mps=step.current_v_mps,
                        )
                        tolerance = (
                            0.0
                            if cfg.maneuver_constraint == "exact"
                            else cfg.maneuver_preference_tolerance_seconds
                        )
                        if best_arrival is None:
                            best_arrival = arrival
                        elif arrival.elapsed_seconds < best_arrival.elapsed_seconds - tolerance:
                            best_arrival = arrival
                        elif abs(arrival.elapsed_seconds - best_arrival.elapsed_seconds) <= tolerance:
                            if arrival.maneuvers < best_arrival.maneuvers or (
                                arrival.maneuvers == best_arrival.maneuvers
                                and arrival.elapsed_seconds < best_arrival.elapsed_seconds
                            ):
                                best_arrival = arrival
                    # A mark crossing completes the leg immediately. It cannot be
                    # sailed through merely to accumulate an exact count.
                    continue

                elapsed = state.elapsed_seconds + penalty + cfg.dt_seconds
                if elapsed > cfg.max_leg_time_seconds:
                    continue
                child = _Node(
                    step.end[0],
                    step.end[1],
                    step.end_time,
                    elapsed,
                    side,
                    maneuvers,
                    state.distance_m + segment_length,
                    (0 if switched else state.moves_on_side) + 1,
                    state,
                    step.heading_deg,
                    step.speed_mps,
                    step.wind,
                    step.current_u_mps,
                    step.current_v_mps,
                )
                key = _node_key(child, cfg.spatial_bin_m)
                incumbent = deduplicated.get(key)
                if incumbent is None or (child.elapsed_seconds, distance(child.position, target)) < (
                    incumbent.elapsed_seconds,
                    distance(incumbent.position, target),
                ):
                    deduplicated[key] = child

        if best_arrival is not None:
            # Every later generation costs at least another full time step; the
            # current best is therefore final when its time is within that bound.
            minimum_frontier_time = min(
                (n.elapsed_seconds for n in deduplicated.values()), default=float("inf")
            )
            tolerance = (
                0.0
                if cfg.maneuver_constraint == "exact"
                else cfg.maneuver_preference_tolerance_seconds
            )
            if minimum_frontier_time >= best_arrival.elapsed_seconds + tolerance:
                return _reconstruct(best_arrival, mode, states_evaluated)

        rank_key = lambda n: (
            n.elapsed_seconds + distance(n.position, target) / cfg.heuristic_speed_mps,
            n.maneuvers,
        )
        ranked = sorted(deduplicated.values(), key=rank_key)
        required_maneuvers = (
            max_maneuvers
            if cfg.maneuver_constraint == "exact"
            else cfg.minimum_maneuvers
        )
        if required_maneuvers > 0 and len(ranked) > cfg.beam_width:
            # A global beam tends to discard states that have already paid the
            # distance/time needed for required maneuvers. Reserve part of the
            # beam for every maneuver-count class, then fill unused capacity
            # with the globally fastest remaining states.
            class_count = max_maneuvers + 1
            quota = max(1, cfg.beam_width // class_count)
            selected: list[_Node] = []
            selected_ids: set[int] = set()
            for maneuver_count in range(class_count):
                members = [n for n in ranked if n.maneuvers == maneuver_count]
                for node in members[:quota]:
                    selected.append(node)
                    selected_ids.add(id(node))
            if len(selected) < cfg.beam_width:
                selected.extend(
                    node
                    for node in ranked
                    if id(node) not in selected_ids
                )
            frontier = sorted(selected, key=rank_key)[: cfg.beam_width]
        else:
            frontier = ranked[: cfg.beam_width]
        if not frontier:
            break

    if best_arrival is not None:
        return _reconstruct(best_arrival, mode, states_evaluated)
    raise RouteNotFoundError(
        f"no {mode} route reached the mark within {cfg.max_leg_time_seconds:.0f} seconds"
    )


def route_race(
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    start_time: datetime,
    wind: WindField,
    speed_model: SpeedModel,
    max_tacks: int = 4,
    max_gybes: int = 4,
    minimum_tacks: int = 0,
    minimum_gybes: int = 0,
    config: RoutingConfig | None = None,
    tack_penalty_seconds: float = 0.0,
    gybe_penalty_seconds: float = 0.0,
    target_model: TargetModel | None = None,
    current: CurrentField | None = None,
) -> RaceResult:
    if minimum_tacks < 0 or minimum_gybes < 0:
        raise ValueError("minimum maneuver counts cannot be negative")
    if minimum_tacks > max_tacks or minimum_gybes > max_gybes:
        raise ValueError("minimum maneuver count cannot exceed maximum")
    cfg = config or RoutingConfig()
    upwind_config = replace(
        cfg,
        maneuver_penalty_seconds=tack_penalty_seconds,
        minimum_maneuvers=minimum_tacks,
    )
    upwind_baselines = (
        one_maneuver_options(
            downwind_mark, upwind_mark, start_time, wind, speed_model,
            "upwind", upwind_config, target_model, current,
        )
        if max_tacks >= 1 else ()
    )
    upwind_candidates: list[RouteResult] = [
        option.route for option in upwind_baselines
        if (
            (upwind_config.maneuver_constraint == "maximum" and option.route.maneuvers_used >= minimum_tacks)
            or (upwind_config.maneuver_constraint == "exact" and max_tacks == 1)
        )
    ]
    try:
        upwind_candidates.append(
            route_leg(
                downwind_mark,
                upwind_mark,
                start_time,
                wind,
                speed_model,
                mode="upwind",
                max_maneuvers=max_tacks,
                config=upwind_config,
                target_model=target_model,
                current=current,
            )
        )
    except RouteNotFoundError:
        if not upwind_candidates:
            raise
    upwind = min(upwind_candidates, key=lambda route: route.elapsed_seconds)

    downwind_config = replace(
        cfg,
        maneuver_penalty_seconds=gybe_penalty_seconds,
        minimum_maneuvers=minimum_gybes,
    )
    downwind_baselines = (
        one_maneuver_options(
            upwind.points[-1].position, downwind_mark, upwind.arrival_time,
            wind, speed_model, "downwind", downwind_config, target_model, current,
        )
        if max_gybes >= 1 else ()
    )
    downwind_candidates: list[RouteResult] = [
        option.route for option in downwind_baselines
        if (
            (downwind_config.maneuver_constraint == "maximum" and option.route.maneuvers_used >= minimum_gybes)
            or (downwind_config.maneuver_constraint == "exact" and max_gybes == 1)
        )
    ]
    try:
        downwind_candidates.append(
            route_leg(
                upwind.points[-1].position,
                downwind_mark,
                upwind.arrival_time,
                wind,
                speed_model,
                mode="downwind",
                max_maneuvers=max_gybes,
                config=downwind_config,
                target_model=target_model,
                current=current,
            )
        )
    except RouteNotFoundError:
        if not downwind_candidates:
            raise
    downwind = min(downwind_candidates, key=lambda route: route.elapsed_seconds)
    return RaceResult(upwind, downwind, upwind_baselines, downwind_baselines)


def route_laps(
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    start_time: datetime,
    wind: WindField,
    speed_model: SpeedModel,
    laps: int = 2,
    max_tacks: int = 4,
    max_gybes: int = 4,
    minimum_tacks: int = 0,
    minimum_gybes: int = 0,
    config: RoutingConfig | None = None,
    tack_penalty_seconds: float = 0.0,
    gybe_penalty_seconds: float = 0.0,
    target_model: TargetModel | None = None,
    current: CurrentField | None = None,
) -> MultiLapResult:
    if laps < 1:
        raise ValueError("laps must be at least one")
    if minimum_tacks < 0 or minimum_gybes < 0:
        raise ValueError("minimum maneuver counts cannot be negative")
    if minimum_tacks > max_tacks or minimum_gybes > max_gybes:
        raise ValueError("minimum maneuver count cannot exceed maximum")
    cfg = config or RoutingConfig()
    legs: list[RouteResult] = []
    lap_results: list[RaceResult] = []
    current_position = downwind_mark
    current_time = start_time
    for _ in range(laps):
        lap = route_race(
            current_position,
            upwind_mark,
            current_time,
            wind,
            speed_model,
            max_tacks=max_tacks,
            max_gybes=max_gybes,
            minimum_tacks=minimum_tacks,
            minimum_gybes=minimum_gybes,
            config=cfg,
            tack_penalty_seconds=tack_penalty_seconds,
            gybe_penalty_seconds=gybe_penalty_seconds,
            target_model=target_model,
            current=current,
        )
        lap_results.append(lap)
        legs.extend((lap.upwind, lap.downwind))
        current_position = lap.downwind.points[-1].position
        current_time = lap.downwind.arrival_time
    return MultiLapResult(tuple(legs), laps, tuple(lap_results))
