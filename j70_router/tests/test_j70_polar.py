from __future__ import annotations

import unittest

from j70_router.sailing.j70_polar import J70Polar, J70SpeedModel
from j70_router.sailing.polar import KNOT_TO_MPS
from j70_router.sailing.targets import expected_vmg_knots, target_twa


class J70PolarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.polar = J70Polar()

    def test_published_table_point_is_preserved(self) -> None:
        self.assertAlmostEqual(self.polar.boat_speed(10.0, 90.0), 6.72, places=6)
        self.assertAlmostEqual(self.polar.boat_speed(12.0, 38.1), 5.718, places=6)

    def test_interpolates_in_tws_and_twa(self) -> None:
        value = self.polar.boat_speed(11.0, 90.0)
        self.assertAlmostEqual(value, (6.72 + 7.03) / 2.0, places=6)
        self.assertGreater(self.polar.boat_speed(10.0, 65.0), 6.17)
        self.assertLess(self.polar.boat_speed(10.0, 65.0), 6.32)

    def test_clamps_wind_and_mirrors_angle(self) -> None:
        self.assertEqual(self.polar.boat_speed(2.0, 90.0), self.polar.boat_speed(6.0, 90.0))
        self.assertEqual(self.polar.boat_speed(30.0, 90.0), self.polar.boat_speed(20.0, 90.0))
        self.assertEqual(self.polar.boat_speed(10.0, -90.0), self.polar.boat_speed(10.0, 90.0))
        self.assertEqual(self.polar.boat_speed(10.0, 270.0), self.polar.boat_speed(10.0, 90.0))

    def test_crew_factor_scales_speed_only(self) -> None:
        model = J70SpeedModel(self.polar, crew_speed_factor=0.93)
        expected = self.polar.boat_speed(10.0, 90.0) * 0.93 * KNOT_TO_MPS
        self.assertAlmostEqual(model(10.0 * KNOT_TO_MPS, 90.0), expected, places=9)

    def test_robust_target_is_derived_from_expected_vmg(self) -> None:
        theoretical = target_twa(self.polar, 10.0, "upwind", "theoretical")
        robust = target_twa(self.polar, 10.0, "upwind", "robust", 3.0)
        self.assertAlmostEqual(theoretical, 40.2, places=6)
        self.assertGreater(robust, theoretical)
        self.assertGreaterEqual(
            expected_vmg_knots(self.polar, 10.0, robust, "upwind", 3.0),
            expected_vmg_knots(self.polar, 10.0, theoretical, "upwind", 3.0),
        )

    def test_asymmetric_spinnaker_planing_transition_is_preserved(self) -> None:
        self.assertIn("asymmetric spinnaker", self.polar.metadata["sail_configuration"])
        self.assertAlmostEqual(self.polar.boat_speed(16.0, 135.0), 9.32, places=6)
        self.assertAlmostEqual(self.polar.boat_speed(20.0, 135.0), 12.68, places=6)
        self.assertAlmostEqual(target_twa(self.polar, 14.0, "downwind", "theoretical"), 144.8)
        self.assertAlmostEqual(target_twa(self.polar, 16.0, "downwind", "theoretical"), 143.7)
        self.assertAlmostEqual(target_twa(self.polar, 20.0, "downwind", "theoretical"), 138.8)
        robust = target_twa(self.polar, 20.0, "downwind", "robust", 3.0)
        self.assertLessEqual(abs(robust - 138.8), 3.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
