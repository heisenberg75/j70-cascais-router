from __future__ import annotations

import unittest

from j70_router.sailing.geometry import LocalCartesian, legal_heading, segment_circle_entry_fraction
from j70_router.wind.base import WindSample


class WindConventionTests(unittest.TestCase):
    def test_northerly_wind_flows_south(self) -> None:
        wind = WindSample.from_speed_direction(10.0, 0.0)
        self.assertAlmostEqual(wind.u_mps, 0.0, places=10)
        self.assertAlmostEqual(wind.v_mps, -10.0, places=10)
        self.assertAlmostEqual(wind.direction_deg, 0.0, places=10)

    def test_easterly_wind_flows_west(self) -> None:
        wind = WindSample.from_speed_direction(10.0, 90.0)
        self.assertAlmostEqual(wind.u_mps, -10.0, places=10)
        self.assertAlmostEqual(wind.v_mps, 0.0, places=10)
        self.assertAlmostEqual(wind.direction_deg, 90.0, places=10)

    def test_westerly_uv_vector_reports_270_degrees(self) -> None:
        wind = WindSample(u_mps=7.0, v_mps=0.0)
        self.assertAlmostEqual(wind.direction_deg, 270.0, places=10)

    def test_legal_headings_use_wind_from_direction(self) -> None:
        self.assertEqual(legal_heading(0.0, "upwind", "port"), 315.0)
        self.assertEqual(legal_heading(0.0, "upwind", "starboard"), 45.0)
        self.assertEqual(legal_heading(0.0, "downwind", "port"), 225.0)
        self.assertEqual(legal_heading(0.0, "downwind", "starboard"), 135.0)


class GeometryTests(unittest.TestCase):
    def test_local_frame_round_trip(self) -> None:
        frame = LocalCartesian(38.695, -9.420)
        expected = (38.7073, -9.4371)
        actual = frame.to_latlon(*frame.to_xy(*expected))
        self.assertAlmostEqual(actual[0], expected[0], places=10)
        self.assertAlmostEqual(actual[1], expected[1], places=10)

    def test_segment_circle_detects_entry_between_steps(self) -> None:
        fraction = segment_circle_entry_fraction((0.0, 0.0), (100.0, 0.0), (80.0, 0.0), 10.0)
        self.assertAlmostEqual(fraction or 0.0, 0.7)


if __name__ == "__main__":
    unittest.main()
