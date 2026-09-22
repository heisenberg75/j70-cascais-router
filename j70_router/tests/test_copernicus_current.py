from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from j70_router.current.copernicus import DATASET_ID, download_current_box, load_cached_current
from j70_router.sailing.geometry import LocalCartesian
from j70_router.ui.map_view import build_course_map
from j70_router.wind.synthetic import SyntheticWindField


UTC = timezone.utc


def remote_dataset() -> xr.Dataset:
    times = np.asarray(
        ["2026-09-21T15:00", "2026-09-21T16:00", "2026-09-21T17:00", "2026-09-21T18:00"],
        dtype="datetime64[ns]",
    )
    latitudes = np.asarray([38.50, 38.70])
    longitudes = np.asarray([-9.70, -9.30])
    u = np.empty((4, 2, 2), dtype=float)
    for index in range(4):
        u[index] = float(index)
    v = np.zeros_like(u)
    return xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), u, {"standard_name": "eastward_sea_water_velocity"}),
            "vo": (("time", "latitude", "longitude"), v, {"standard_name": "northward_sea_water_velocity"}),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )


class CopernicusCurrentTests(unittest.TestCase):
    def test_coastal_land_corner_does_not_poison_wet_interpolation(self) -> None:
        dataset = remote_dataset()
        dataset["uo"].values[:, 0, 0] = np.nan
        dataset["vo"].values[:, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            source = download_current_box(
                38.5,
                38.7,
                -9.7,
                -9.3,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                cache_dir=directory,
                opener=lambda **kwargs: dataset,
            )
            value = source.current_at(
                38.6, -9.5, datetime(2026, 9, 21, 16, tzinfo=UTC)
            )
        self.assertTrue(np.isfinite(value["u_ms"]))
        self.assertTrue(np.isfinite(value["v_ms"]))
        self.assertAlmostEqual(value["u_ms"], 1.0)

    def test_subset_cache_interpolation_and_direction_to(self) -> None:
        calls = []

        def opener(**kwargs):
            calls.append(kwargs)
            return remote_dataset()

        with tempfile.TemporaryDirectory() as directory:
            source = download_current_box(
                38.5,
                38.7,
                -9.7,
                -9.3,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                cache_dir=directory,
                opener=opener,
            )
            self.assertEqual(calls[0]["dataset_id"], DATASET_ID)
            self.assertEqual(calls[0]["variables"], ["uo", "vo"])
            value = source.current_at(
                38.6, -9.5, datetime(2026, 9, 21, 16, 30, tzinfo=UTC)
            )
            self.assertAlmostEqual(value["u_ms"], 1.5)
            self.assertAlmostEqual(value["v_ms"], 0.0)
            self.assertAlmostEqual(value["direction_to_deg"], 90.0)
            cached = load_cached_current(directory)
            self.assertEqual(cached.grid.times, source.grid.times)

    def test_current_map_has_raster_arrows_and_current_legend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = download_current_box(
                38.5,
                38.7,
                -9.7,
                -9.3,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                cache_dir=directory,
                opener=lambda **kwargs: remote_dataset(),
            )
            frame = LocalCartesian(38.6, -9.5)
            leeward, windward = (0.0, 0.0), (0.0, 1000.0)
            html = build_course_map(
                frame.to_latlon(*leeward),
                frame.to_latlon(*windward),
                frame,
                leeward,
                windward,
                SyntheticWindField.uniform(5.0, 0.0),
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                show_wind=False,
                current_source=source,
                current_time=datetime(2026, 9, 21, 16, tzinfo=UTC),
            ).get_root().render()
            self.assertIn("Surface current [kt]", html)
            self.assertIn("Current arrows", html)
            self.assertIn("data:image/png;base64", html)


if __name__ == "__main__":
    unittest.main()
