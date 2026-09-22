"""Run the requested 2026-09-21 16:00 UTC two-lap Cascais test."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from math import cos, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import plot_multilap_and_wind
from .routing.optimizer import MultiLapResult, RoutingConfig, route_laps
from .sailing.geometry import LocalCartesian, heading_vector
from .sailing.polar import ConstantSpeedPolar
from .wind.gridded import RegularGridWindField

START_TIME = datetime(2026, 9, 21, 16, 0, tzinfo=timezone.utc)
TARGET_DURATION_SECONDS = 90.0 * 60.0
COURSE_CENTRE = (38.6390, -9.4970)
FORECAST_PATH = Path("data/forecast/icon_eu_cascais_20260921.json")
OUTPUT_PLOT = Path("artifacts/today_1600_two_laps.png")
OUTPUT_SAMPLES = Path("artifacts/today_1600_route_wind.csv")
OUTPUT_SUMMARY = Path("artifacts/today_1600_summary.json")


def _marks(axis_heading_deg: float, length_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    east, north = heading_vector(axis_heading_deg)
    half = length_m / 2.0
    return (-east * half, -north * half), (east * half, north * half)


def _run_for_length(
    length_m: float,
    axis_heading_deg: float,
    wind: RegularGridWindField,
    config: RoutingConfig,
) -> tuple[tuple[float, float], tuple[float, float], MultiLapResult]:
    downwind_mark, upwind_mark = _marks(axis_heading_deg, length_m)
    race = route_laps(
        downwind_mark,
        upwind_mark,
        START_TIME,
        wind,
        ConstantSpeedPolar(6.0),
        laps=2,
        max_tacks=4,
        max_gybes=4,
        config=config,
    )
    return downwind_mark, upwind_mark, race


def _fit_course(
    axis_heading_deg: float,
    wind: RegularGridWindField,
    config: RoutingConfig,
) -> tuple[float, tuple[float, float], tuple[float, float], MultiLapResult]:
    # Constant-speed first estimate: four legs, each sailed at 45 degrees.
    length_m = TARGET_DURATION_SECONDS * (6.0 * 0.514444) * cos(radians(45.0)) / 4.0
    result = None
    for _ in range(5):
        downwind_mark, upwind_mark, race = _run_for_length(length_m, axis_heading_deg, wind, config)
        result = downwind_mark, upwind_mark, race
        error = TARGET_DURATION_SECONDS - race.total_elapsed_seconds
        if abs(error) <= 5.0:
            break
        length_m *= TARGET_DURATION_SECONDS / race.total_elapsed_seconds
    assert result is not None
    return length_m, result[0], result[1], result[2]


def _write_route_samples(
    race: MultiLapResult,
    wind: RegularGridWindField,
    frame: LocalCartesian,
) -> None:
    OUTPUT_SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SAMPLES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lap", "leg", "mode", "time_utc", "latitude", "longitude", "side",
                "wind_speed_mps", "wind_speed_knots", "wind_from_deg", "boat_speed_mps",
            ],
        )
        writer.writeheader()
        for leg_index, leg in enumerate(race.legs):
            for point in leg.points:
                lat, lon = frame.to_latlon(point.x, point.y)
                sample = wind.wind_at(point.position, point.time)
                writer.writerow(
                    {
                        "lap": leg_index // 2 + 1,
                        "leg": leg_index + 1,
                        "mode": leg.mode,
                        "time_utc": point.time.isoformat(),
                        "latitude": f"{lat:.7f}",
                        "longitude": f"{lon:.7f}",
                        "side": point.side,
                        "wind_speed_mps": f"{sample.speed_mps:.3f}",
                        "wind_speed_knots": f"{sample.speed_mps / 0.514444:.2f}",
                        "wind_from_deg": f"{sample.direction_deg:.1f}",
                        "boat_speed_mps": "" if point.boat_speed_mps is None else f"{point.boat_speed_mps:.3f}",
                    }
                )


def main() -> None:
    frame = LocalCartesian(*COURSE_CENTRE)
    wind = RegularGridWindField.from_open_meteo_json(FORECAST_PATH, frame)
    initial = wind.wind_at((0.0, 0.0), START_TIME)
    axis_heading = initial.direction_deg
    config = RoutingConfig(
        dt_seconds=10.0,
        spatial_bin_m=35.0,
        mark_radius_m=15.0,
        beam_width=8_000,
        max_leg_time_seconds=3_600.0,
    )
    length_m, downwind_mark, upwind_mark, race = _fit_course(axis_heading, wind, config)
    midpoint_time = START_TIME + timedelta(seconds=race.total_elapsed_seconds / 2.0)
    figure = plot_multilap_and_wind(
        race,
        downwind_mark,
        upwind_mark,
        wind,
        midpoint_time,
        OUTPUT_PLOT,
    )
    plt.close(figure)
    _write_route_samples(race, wind, frame)

    down_latlon = frame.to_latlon(*downwind_mark)
    up_latlon = frame.to_latlon(*upwind_mark)
    speeds = []
    directions = []
    for leg in race.legs:
        for point in leg.points:
            sample = wind.wind_at(point.position, point.time)
            speeds.append(sample.speed_mps)
            directions.append(sample.direction_deg)
    leg_summaries = []
    for leg_index, leg in enumerate(race.legs):
        maneuvers = []
        for maneuver in leg.maneuvers:
            lat, lon = frame.to_latlon(maneuver.x, maneuver.y)
            maneuvers.append(
                {
                    "time_utc": maneuver.time.isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "from_side": maneuver.from_side,
                    "to_side": maneuver.to_side,
                }
            )
        leg_summaries.append(
            {
                "lap": leg_index // 2 + 1,
                "mode": leg.mode,
                "elapsed_seconds": leg.elapsed_seconds,
                "distance_m": leg.distance_m,
                "maneuvers_used": leg.maneuvers_used,
                "arrival_time_utc": leg.arrival_time.isoformat(),
                "maneuvers": maneuvers,
            }
        )
    summary = {
        "forecast_source": wind.metadata,
        "race_start_utc": START_TIME.isoformat(),
        "laps": race.laps,
        "course_centre_latlon": COURSE_CENTRE,
        "course_axis_heading_deg": axis_heading,
        "course_length_m": length_m,
        "downwind_mark_latlon": down_latlon,
        "upwind_mark_latlon": up_latlon,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "routed_duration_seconds": race.total_elapsed_seconds,
        "finish_time_utc": race.arrival_time.isoformat(),
        "total_distance_m": race.total_distance_m,
        "tacks": race.tacks,
        "gybes": race.gybes,
        "legs": leg_summaries,
        "route_wind_speed_mps_min": min(speeds),
        "route_wind_speed_mps_max": max(speeds),
        "route_wind_direction_deg_min": min(directions),
        "route_wind_direction_deg_max": max(directions),
        "constant_boat_speed_knots": 6.0,
        "plot": str(OUTPUT_PLOT),
        "route_samples": str(OUTPUT_SAMPLES),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
