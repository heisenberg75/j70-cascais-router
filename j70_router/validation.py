"""Generate visual synthetic validations A-D without external data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import plot_race_and_wind
from .routing.optimizer import RoutingConfig, route_race
from .sailing.polar import ConstantSpeedPolar, WindScaledTestPolar
from .wind.synthetic import SyntheticWindField


def _render(
    name: str,
    wind: SyntheticWindField,
    start_time: datetime,
    speed_model,
    output_dir: Path,
) -> None:
    config = RoutingConfig(
        dt_seconds=10.0,
        spatial_bin_m=25.0,
        mark_radius_m=15.0,
        beam_width=6_000,
        max_leg_time_seconds=3_600.0,
    )
    race = route_race(
        (0.0, 0.0),
        (0.0, 1000.0),
        start_time,
        wind,
        speed_model,
        max_tacks=3,
        max_gybes=3,
        config=config,
    )
    figure = plot_race_and_wind(
        race,
        (0.0, 0.0),
        (0.0, 1000.0),
        wind,
        start_time,
        output_dir / f"{name}.png",
    )
    plt.close(figure)
    print(
        f"{name}: {race.total_elapsed_seconds / 60:.2f} min, "
        f"{race.upwind.maneuvers_used} tacks, {race.downwind.maneuvers_used} gybes"
    )


def main() -> None:
    output_dir = Path("artifacts/validation")
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    _render(
        "test_a_uniform_northerly",
        SyntheticWindField.uniform(8.0, 0.0),
        start,
        ConstantSpeedPolar(),
        output_dir,
    )
    _render(
        "test_b_western_pressure",
        SyntheticWindField(
            base_speed_mps=7.0,
            base_direction_deg=0.0,
            speed_gradient_x_per_m=-0.003,
            reference_time=start,
        ),
        start,
        WindScaledTestPolar(),
        output_dir,
    )
    _render(
        "test_c_cross_course_shift",
        SyntheticWindField(
            base_speed_mps=8.0,
            base_direction_deg=0.0,
            direction_gradient_x_deg_per_m=0.02,
            reference_time=start,
        ),
        start,
        ConstantSpeedPolar(),
        output_dir,
    )
    changing = SyntheticWindField(
        base_speed_mps=8.0,
        base_direction_deg=350.0,
        direction_rate_deg_per_hour=20.0,
        reference_time=start,
    )
    _render("test_d_time_shift_early", changing, start, ConstantSpeedPolar(), output_dir)
    _render(
        "test_d_time_shift_late",
        changing,
        start + timedelta(hours=2),
        ConstantSpeedPolar(),
        output_dir,
    )


if __name__ == "__main__":
    main()
