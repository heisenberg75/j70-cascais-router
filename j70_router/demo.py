"""Generate the first-milestone Cascais synthetic route plot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .plotting import plot_race_and_wind
from .routing.optimizer import RoutingConfig, route_race
from .sailing.geometry import LocalCartesian
from .sailing.polar import ConstantSpeedPolar
from .wind.synthetic import SyntheticWindField


def _duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{secs:04.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestone_route.png"))
    parser.add_argument("--speed-knots", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--max-tacks", type=int, default=4)
    parser.add_argument("--max-gybes", type=int, default=4)
    args = parser.parse_args()

    downwind_latlon = (38.6950, -9.4200)
    upwind_latlon = (38.7085, -9.4200)
    frame = LocalCartesian(*downwind_latlon)
    downwind_mark = frame.to_xy(*downwind_latlon)
    upwind_mark = frame.to_xy(*upwind_latlon)
    start_time = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    wind = SyntheticWindField.uniform(speed_mps=8.0, direction_deg=0.0)
    config = RoutingConfig(dt_seconds=args.dt, spatial_bin_m=25.0, mark_radius_m=15.0)
    race = route_race(
        downwind_mark,
        upwind_mark,
        start_time,
        wind,
        ConstantSpeedPolar(args.speed_knots),
        max_tacks=args.max_tacks,
        max_gybes=args.max_gybes,
        config=config,
    )
    plot_race_and_wind(race, downwind_mark, upwind_mark, wind, start_time, args.output)

    print(f"Plot: {args.output.resolve()}")
    print(f"Total race time: {_duration(race.total_elapsed_seconds)}")
    print(f"Upwind time: {_duration(race.upwind.elapsed_seconds)}")
    print(f"Downwind time: {_duration(race.downwind.elapsed_seconds)}")
    print(f"Upwind distance: {race.upwind.distance_m:.1f} m")
    print(f"Downwind distance: {race.downwind.distance_m:.1f} m")
    print(f"Total distance: {race.total_distance_m:.1f} m")
    print(f"Tacks: {race.upwind.maneuvers_used}")
    print(f"Gybes: {race.downwind.maneuvers_used}")
    for label, maneuvers in (("Tack", race.upwind.maneuvers), ("Gybe", race.downwind.maneuvers)):
        for number, maneuver in enumerate(maneuvers, 1):
            lat, lon = frame.to_latlon(maneuver.x, maneuver.y)
            print(f"{label} {number}: {lat:.6f}, {lon:.6f} at {maneuver.time.isoformat()}")


if __name__ == "__main__":
    main()
