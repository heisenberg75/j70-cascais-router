from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from j70_router.sailing.geometry import LocalCartesian
from j70_router.wind.gridded import RegularGridWindField


class RegularGridWindTests(unittest.TestCase):
    def test_bilinear_space_and_linear_uv_time_interpolation(self) -> None:
        frame = LocalCartesian(0.0, 0.0)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        u = np.array(
            [
                [[0.0, 2.0], [2.0, 4.0]],
                [[10.0, 12.0], [12.0, 14.0]],
            ]
        )
        v = np.array(
            [
                [[0.0, 4.0], [4.0, 8.0]],
                [[20.0, 24.0], [24.0, 28.0]],
            ]
        )
        field = RegularGridWindField(
            frame,
            np.array([-0.01, 0.01]),
            np.array([-0.01, 0.01]),
            (start, start + timedelta(hours=1)),
            u,
            v,
        )
        sample = field.wind_at((0.0, 0.0), start + timedelta(minutes=30))
        self.assertAlmostEqual(sample.u_mps, 7.0)
        self.assertAlmostEqual(sample.v_mps, 14.0)


if __name__ == "__main__":
    unittest.main()
