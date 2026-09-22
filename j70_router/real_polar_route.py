"""One-lap real-polar route using the available Cascais forecast grid."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import plot_race_and_wind
from .routing.optimizer import RoutingConfig, route_race
from .sailing.geometry import LocalCartesian, heading_vector
from .sailing.j70_polar import J70Polar, J70SpeedModel
from .sailing.targets import TargetAngleModel
from .wind.gridded import RegularGridWindField

START_TIME = datetime(2026, 9, 21, 16, tzinfo=timezone.utc)
COURSE_CENTRE = (38.6390, -9.4970)
COURSE_LENGTH_M = 1500.0
FORECAST_PATH = Path("data/forecast/icon_eu_cascais_20260921.json")
OUTPUT_PLOT = Path("artifacts/real_polar_route_icon_fallback.png")
OUTPUT_SUMMARY = Path("artifacts/real_polar_route_icon_fallback.json")


def main() -> None:
    frame = LocalCartesian(*COURSE_CENTRE)
    wind = RegularGridWindField.from_open_meteo_json(FORECAST_PATH, frame)
    initial = wind.wind_at((0.0, 0.0), START_TIME)
    east, north = heading_vector(initial.direction_deg)
    half = COURSE_LENGTH_M / 2.0
    downwind_mark = (-east * half, -north * half)
    upwind_mark = (east * half, north * half)

    polar = J70Polar()
    speed_model = J70SpeedModel(polar, crew_speed_factor=0.93)
    target_model = TargetAngleModel(polar, sailing_mode="robust", heading_sigma_deg=3.0)
    race = route_race(
        downwind_mark,
        upwind_mark,
        START_TIME,
        wind,
        speed_model,
        max_tacks=4,
        max_gybes=4,
        target_model=target_model,
        config=RoutingConfig(
            dt_seconds=10.0,
            spatial_bin_m=30.0,
            mark_radius_m=15.0,
            beam_width=8_000,
            max_leg_time_seconds=3_600.0,
        ),
    )
    figure = plot_race_and_wind(
        race, downwind_mark, upwind_mark, wind, START_TIME, OUTPUT_PLOT, padding_m=350
    )
    figure.suptitle(
        f"J/70 robust-polar route (ICON-EU fallback) — {race.total_elapsed_seconds / 60:.1f} min"
    )
    figure.savefig(OUTPUT_PLOT, dpi=170, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "forecast_source": wind.metadata,
        "weather_status": "ICON-EU fallback; live GribStream AROME awaits GRIBSTREAM_API_KEY",
        "race_start_utc": START_TIME.isoformat(),
        "course_centre_latlon": COURSE_CENTRE,
        "course_length_m": COURSE_LENGTH_M,
        "downwind_mark_latlon": frame.to_latlon(*downwind_mark),
        "upwind_mark_latlon": frame.to_latlon(*upwind_mark),
        "sailing_mode": "robust",
        "heading_sigma_deg": 3.0,
        "crew_speed_factor": 0.93,
        "elapsed_seconds": race.total_elapsed_seconds,
        "upwind_seconds": race.upwind.elapsed_seconds,
        "downwind_seconds": race.downwind.elapsed_seconds,
        "distance_m": race.total_distance_m,
        "tacks": race.upwind.maneuvers_used,
        "gybes": race.downwind.maneuvers_used,
        "finish_time_utc": race.downwind.arrival_time.isoformat(),
        "initial_wind_speed_knots": initial.speed_mps / 0.514444,
        "initial_wind_from_deg": initial.direction_deg,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
