from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter

from j70_router.routing.optimizer import (
    RouteNotFoundError,
    RoutingConfig,
    adaptive_routing_config,
    heading_sector_margin_deg,
    route_laps,
    route_leg,
    route_race,
)
from j70_router.sailing.polar import ConstantSpeedPolar, WindScaledTestPolar
from j70_router.wind.synthetic import SyntheticWindField


START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RoutingConfig(
            dt_seconds=10.0,
            spatial_bin_m=25.0,
            mark_radius_m=15.0,
            beam_width=6000,
            max_leg_time_seconds=3600.0,
        )

    def test_infeasible_route_respects_computation_timeout(self) -> None:
        started = perf_counter()
        with self.assertRaisesRegex(RouteNotFoundError, "search stopped"):
            route_leg(
                (0.0, 0.0),
                (0.0, -1500.0),
                START,
                SyntheticWindField.uniform(5.0, 0.0),
                ConstantSpeedPolar(),
                mode="upwind",
                max_maneuvers=4,
                config=RoutingConfig(
                    dt_seconds=5,
                    spatial_bin_m=20,
                    mark_radius_m=15,
                    beam_width=8000,
                    max_leg_time_seconds=7200,
                    max_computation_seconds=0.02,
                ),
            )
        self.assertLess(perf_counter() - started, 1.0)

    def test_heading_sector_diagnostic_detects_reversed_marks(self) -> None:
        self.assertGreater(
            heading_sector_margin_deg((0, 0), (0, 1000), 0, 45, "upwind"), 0
        )
        self.assertLess(
            heading_sector_margin_deg((0, 0), (0, -1000), 0, 45, "upwind"), 0
        )
        self.assertGreater(
            heading_sector_margin_deg((0, 1000), (0, 0), 0, 135, "downwind"), 0
        )
        self.assertLess(
            heading_sector_margin_deg((0, 0), (0, 1000), 0, 135, "downwind"), 0
        )

    def test_long_course_adapts_search_without_reducing_requested_fidelity(self) -> None:
        base = RoutingConfig(
            dt_seconds=10,
            spatial_bin_m=30,
            beam_width=2500,
            max_leg_time_seconds=7200,
            max_computation_seconds=8,
        )
        short = adaptive_routing_config(base, 1500)
        long = adaptive_routing_config(base, 9000)
        self.assertEqual(short.dt_seconds, 10)
        self.assertGreaterEqual(long.dt_seconds, 20)
        self.assertGreaterEqual(long.spatial_bin_m, long.dt_seconds * 3)
        self.assertLessEqual(long.beam_width, 400)
        self.assertGreater(long.max_leg_time_seconds, 7200)
        self.assertGreater(long.max_computation_seconds, 8)

    def test_uniform_northerly_one_lap_reaches_both_marks(self) -> None:
        wind = SyntheticWindField.uniform(8.0, 0.0)
        race = route_race(
            (0.0, 0.0),
            (0.0, 1000.0),
            START,
            wind,
            ConstantSpeedPolar(6.0),
            max_tacks=4,
            max_gybes=4,
            config=self.config,
        )
        self.assertLessEqual(race.upwind.maneuvers_used, 4)
        self.assertLessEqual(race.downwind.maneuvers_used, 4)
        self.assertLessEqual(abs(race.upwind.points[-1].x), self.config.mark_radius_m)
        self.assertLessEqual(abs(race.downwind.points[-1].x), self.config.mark_radius_m)
        self.assertGreater(race.total_elapsed_seconds, 0.0)
        self.assertEqual(
            {option.initial_side for option in race.upwind_baselines},
            {"port", "starboard"},
        )
        self.assertEqual(
            {option.course_side for option in race.upwind_baselines},
            {"left", "right"},
        )

    def test_favourable_current_changes_ground_track_and_arrival(self) -> None:
        class NorthCurrent:
            def current_at(self, position, time):
                return 0.0, 0.5

        wind = SyntheticWindField.uniform(8.0, 0.0)
        without = route_leg(
            (0.0, 0.0), (0.0, 1000.0), START, wind, ConstantSpeedPolar(6.0),
            mode="upwind", max_maneuvers=2, config=self.config,
        )
        with_current = route_leg(
            (0.0, 0.0), (0.0, 1000.0), START, wind, ConstantSpeedPolar(6.0),
            mode="upwind", max_maneuvers=2, config=self.config, current=NorthCurrent(),
        )
        self.assertLess(with_current.elapsed_seconds, without.elapsed_seconds)
        self.assertTrue(any(point.current_v_mps == 0.5 for point in with_current.points[1:]))

    def test_two_laps_preserve_arrival_time_between_four_legs(self) -> None:
        wind = SyntheticWindField.uniform(8.0, 0.0)
        race = route_laps(
            (0.0, 0.0),
            (0.0, 500.0),
            START,
            wind,
            ConstantSpeedPolar(),
            laps=2,
            max_tacks=2,
            max_gybes=2,
            config=self.config,
        )
        self.assertEqual(len(race.legs), 4)
        self.assertEqual(len(race.lap_results), 2)
        self.assertEqual([leg.mode for leg in race.legs], ["upwind", "downwind", "upwind", "downwind"])
        self.assertTrue(all(lap.upwind_baselines for lap in race.lap_results))
        self.assertTrue(all(lap.downwind_baselines for lap in race.lap_results))
        for previous, current in zip(race.legs, race.legs[1:]):
            self.assertEqual(previous.arrival_time, current.points[0].time)

    def test_exact_maneuver_count_is_honoured(self) -> None:
        exact = RoutingConfig(**{**self.config.__dict__, "maneuver_constraint": "exact"})
        route = route_leg(
            (0.0, 0.0),
            (0.0, 600.0),
            START,
            SyntheticWindField.uniform(8.0, 0.0),
            ConstantSpeedPolar(6.0),
            mode="upwind",
            max_maneuvers=3,
            config=exact,
        )
        self.assertEqual(route.maneuvers_used, 3)
        self.assertEqual(len(route.maneuvers), 3)

    def test_minimum_maneuvers_and_penalties_are_applied(self) -> None:
        wind = SyntheticWindField.uniform(8.0, 0.0)
        no_penalty = route_race(
            (0.0, 0.0), (0.0, 700.0), START, wind, ConstantSpeedPolar(6.0),
            max_tacks=2, max_gybes=2, minimum_tacks=1, minimum_gybes=1,
            config=self.config,
        )
        with_penalty = route_race(
            (0.0, 0.0), (0.0, 700.0), START, wind, ConstantSpeedPolar(6.0),
            max_tacks=2, max_gybes=2, minimum_tacks=1, minimum_gybes=1,
            tack_penalty_seconds=7.0, gybe_penalty_seconds=11.0,
            config=self.config,
        )
        self.assertGreaterEqual(no_penalty.upwind.maneuvers_used, 1)
        self.assertGreaterEqual(no_penalty.downwind.maneuvers_used, 1)
        self.assertAlmostEqual(
            with_penalty.total_elapsed_seconds - no_penalty.total_elapsed_seconds,
            18.0,
            delta=1e-9,
        )

    def test_minimum_maneuver_separation_is_honoured(self) -> None:
        route = route_leg(
            (0.0, 0.0),
            (0.0, 600.0),
            START,
            SyntheticWindField.uniform(8.0, 0.0),
            ConstantSpeedPolar(6.0),
            mode="upwind",
            max_maneuvers=3,
            config=replace(
                self.config,
                maneuver_constraint="exact",
                minimum_maneuver_separation_seconds=20.0,
            ),
        )
        times = [maneuver.time for maneuver in route.maneuvers]
        self.assertEqual(len(times), 3)
        self.assertTrue(all((right - left).total_seconds() >= 20.0 for left, right in zip(times, times[1:])))

    def test_western_pressure_field_draws_route_west(self) -> None:
        wind = SyntheticWindField(
            base_speed_mps=7.0,
            base_direction_deg=0.0,
            speed_gradient_x_per_m=-0.003,
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        route = route_leg(
            (0.0, 0.0),
            (0.0, 1000.0),
            START,
            wind,
            WindScaledTestPolar(),
            mode="upwind",
            max_maneuvers=2,
            config=replace(self.config, maneuver_preference_tolerance_seconds=0.0),
        )
        self.assertLess(min(point.x for point in route.points), -200.0)

    def test_cross_course_direction_shift_selects_favoured_side(self) -> None:
        wind = SyntheticWindField(
            base_speed_mps=8.0,
            base_direction_deg=0.0,
            direction_gradient_x_deg_per_m=0.02,
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        route = route_leg(
            (0.0, 0.0),
            (0.0, 1000.0),
            START,
            wind,
            ConstantSpeedPolar(),
            mode="upwind",
            max_maneuvers=3,
            config=replace(self.config, maneuver_preference_tolerance_seconds=0.0),
        )
        west_extent = abs(min(point.x for point in route.points))
        east_extent = max(point.x for point in route.points)
        self.assertGreater(abs(west_extent - east_extent), 250.0)

    def test_time_changing_wind_changes_route(self) -> None:
        wind = SyntheticWindField(
            base_speed_mps=8.0,
            base_direction_deg=350.0,
            direction_rate_deg_per_hour=20.0,
            reference_time=START,
        )
        early = route_leg(
            (0.0, 0.0), (0.0, 800.0), START, wind, ConstantSpeedPolar(),
            max_maneuvers=3, config=self.config,
        )
        late_start = START.replace(hour=14)
        late = route_leg(
            (0.0, 0.0), (0.0, 800.0), late_start, wind, ConstantSpeedPolar(),
            max_maneuvers=3, config=self.config,
        )
        early_east_extent = max(point.x for point in early.points)
        late_east_extent = max(point.x for point in late.points)
        self.assertGreater(abs(early_east_extent - late_east_extent), 100.0)
        self.assertNotEqual(early.maneuvers_used, late.maneuvers_used)


if __name__ == "__main__":
    unittest.main()
