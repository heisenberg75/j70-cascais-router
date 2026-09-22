"""Download latest AROME Cascais subset and run one robust J/70 lap."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import plot_race_and_wind
from .routing.optimizer import RoutingConfig, route_race
from .sailing.geometry import LocalCartesian, heading_vector
from .sailing.j70_polar import J70Polar, J70SpeedModel
from .sailing.targets import TargetAngleModel
from .wind.arome import download_wind_box

START_TIME = datetime(2026, 9, 21, 16, tzinfo=timezone.utc)
COURSE_CENTRE = (38.6390, -9.4970)
COURSE_LENGTH_M = 1500.0
OUTPUT_PLOT = Path("artifacts/current_arome_j70_route.png")
OUTPUT_SUMMARY = Path("artifacts/current_arome_j70_route.json")


def main() -> None:
    end_time = START_TIME + timedelta(hours=1.5)
    source = download_wind_box(
        38.57, 38.71, -9.59, -9.40,
        START_TIME, end_time,
    )
    frame = LocalCartesian(*COURSE_CENTRE)
    wind = source.for_local_frame(frame)
    initial = wind.wind_at((0.0, 0.0), START_TIME)
    east, north = heading_vector(initial.direction_deg)
    half = COURSE_LENGTH_M / 2.0
    downwind_mark = (-east * half, -north * half)
    upwind_mark = (east * half, north * half)

    polar = J70Polar()
    race = route_race(
        downwind_mark, upwind_mark, START_TIME, wind,
        J70SpeedModel(polar, crew_speed_factor=0.93),
        max_tacks=4,
        max_gybes=4,
        target_model=TargetAngleModel(polar, "robust", 3.0),
        config=RoutingConfig(
            dt_seconds=10.0, spatial_bin_m=30.0, mark_radius_m=15.0,
            beam_width=8_000, max_leg_time_seconds=3_600.0,
        ),
    )
    figure = plot_race_and_wind(
        race, downwind_mark, upwind_mark, wind, START_TIME, OUTPUT_PLOT, padding_m=350
    )
    figure.suptitle(
        f"Latest AROME J/70 robust route — run {source.forecast_run:%Y-%m-%d %H:%M} UTC"
    )
    figure.savefig(OUTPUT_PLOT, dpi=170, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "model": "aromefrance",
        "forecast_run_utc": source.forecast_run.isoformat(),
        "cache_path": str(source.cache_path),
        "race_start_utc": START_TIME.isoformat(),
        "course_centre_latlon": COURSE_CENTRE,
        "downwind_mark_latlon": frame.to_latlon(*downwind_mark),
        "upwind_mark_latlon": frame.to_latlon(*upwind_mark),
        "elapsed_seconds": race.total_elapsed_seconds,
        "upwind_seconds": race.upwind.elapsed_seconds,
        "downwind_seconds": race.downwind.elapsed_seconds,
        "distance_m": race.total_distance_m,
        "tacks": race.upwind.maneuvers_used,
        "gybes": race.downwind.maneuvers_used,
        "initial_wind": source.wind_at(*COURSE_CENTRE, START_TIME),
        "plot": str(OUTPUT_PLOT),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
