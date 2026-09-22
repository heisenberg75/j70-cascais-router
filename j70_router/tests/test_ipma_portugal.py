from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import xarray as xr

from j70_router.wind.ipma_portugal import (
    _url_exists,
    get_latest_ipma_portugal_run,
    load_ipma_portugal_file,
    run_url,
    run_urls,
)

UTC = timezone.utc


class IpmaPortugalTests(unittest.TestCase):
    def test_access_denial_is_not_misreported_as_missing_run(self) -> None:
        response = Mock(status_code=403)
        with patch("j70_router.wind.ipma_portugal.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "denied access.*HTTP 403"):
                _url_exists("https://mf2.ipma.pt/example.nc")
        response.close.assert_called_once()

    def test_latest_run_probes_newest_published_cycle(self) -> None:
        expected = datetime(2026, 9, 22, 0, tzinfo=UTC)
        seen: list[str] = []

        def exists(url: str) -> bool:
            seen.append(url)
            return url == run_url(expected)

        actual = get_latest_ipma_portugal_run(
            datetime(2026, 9, 22, 10, tzinfo=UTC), exists=exists
        )
        self.assertEqual(actual, expected)
        self.assertIn("PT2_025_UVCOMP_10-HTGL_2026092206.nc", seen[0])
        self.assertTrue(any("PT2_025_UVCOMP_10-HTGL_2026092200.nc" in url for url in seen))

    def test_latest_run_checks_six_hour_operational_cycles(self) -> None:
        expected = datetime(2026, 9, 22, 18, tzinfo=UTC)

        actual = get_latest_ipma_portugal_run(
            datetime(2026, 9, 22, 20, tzinfo=UTC),
            exists=lambda url: url == run_url(expected),
        )

        self.assertEqual(actual, expected)

    def test_latest_run_accepts_uv_file_when_uvcomp_is_absent(self) -> None:
        expected = datetime(2026, 9, 22, 12, tzinfo=UTC)
        uv_url = run_urls(expected)[1]

        actual = get_latest_ipma_portugal_run(
            datetime(2026, 9, 22, 17, tzinfo=UTC),
            exists=lambda url: url == uv_url,
        )

        self.assertEqual(actual, expected)
        self.assertIn("PT2_025_UV_10-HTGL_2026092212.nc", uv_url)

    def test_metadata_driven_uv_detection_and_subset(self) -> None:
        times = np.asarray(["2026-09-22T00:00", "2026-09-22T01:00"], dtype="datetime64[m]")
        latitudes = np.asarray([38.55, 38.60, 38.65, 38.70])
        longitudes = np.asarray([-9.60, -9.55, -9.50, -9.45])
        shape = (len(times), len(latitudes), len(longitudes))
        dataset = xr.Dataset(
            {
                "east_component": (
                    ("time", "latitude", "longitude"),
                    np.full(shape, 3.0),
                    {"standard_name": "eastward_wind", "height": "10 m"},
                ),
                "north_component": (
                    ("time", "latitude", "longitude"),
                    np.full(shape, 4.0),
                    {"standard_name": "northward_wind", "height": "10 m"},
                ),
            },
            coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ipma.nc"
            dataset.to_netcdf(path, engine="h5netcdf")
            source = load_ipma_portugal_file(
                path,
                38.58,
                38.68,
                -9.58,
                -9.48,
                datetime(2026, 9, 22, 0, tzinfo=UTC),
            )
        value = source.wind_at(38.625, -9.525, datetime(2026, 9, 22, 0, 30, tzinfo=UTC))
        self.assertAlmostEqual(value["u_ms"], 3.0)
        self.assertAlmostEqual(value["v_ms"], 4.0)
        self.assertAlmostEqual(value["speed_ms"], 5.0)
        self.assertAlmostEqual(value["direction_from_deg"], 216.8698976)

    def test_initialization_plus_step_uses_forward_valid_times(self) -> None:
        initialization = np.asarray(["2026-09-22T00:00"], dtype="datetime64[m]")
        steps = np.asarray([0, 1, 2], dtype="timedelta64[h]")
        valid_times = initialization[:, None] + steps[None, :]
        latitudes = np.asarray([38.55, 38.60, 38.65, 38.70])
        longitudes = np.asarray([-9.60, -9.55, -9.50, -9.45])
        shape = (1, len(steps), len(latitudes), len(longitudes))
        u = np.empty(shape)
        for index in range(len(steps)):
            u[:, index, :, :] = float(index + 1)
        dataset = xr.Dataset(
            {
                "east_component": (
                    ("time", "step", "latitude", "longitude"),
                    u,
                    {"standard_name": "eastward_wind", "height": "10 m"},
                ),
                "north_component": (
                    ("time", "step", "latitude", "longitude"),
                    np.zeros(shape),
                    {"standard_name": "northward_wind", "height": "10 m"},
                ),
            },
            coords={
                "time": initialization,
                "step": steps,
                "valid_time": (("time", "step"), valid_times),
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ipma_steps.nc"
            dataset.to_netcdf(path, engine="h5netcdf")
            source = load_ipma_portugal_file(
                path,
                38.58,
                38.68,
                -9.58,
                -9.48,
                datetime(2026, 9, 22, 0, tzinfo=UTC),
            )

        self.assertEqual(
            source.grid.times,
            (
                datetime(2026, 9, 22, 0, tzinfo=UTC),
                datetime(2026, 9, 22, 1, tzinfo=UTC),
                datetime(2026, 9, 22, 2, tzinfo=UTC),
            ),
        )
        value = source.wind_at(38.625, -9.525, datetime(2026, 9, 22, 1, 30, tzinfo=UTC))
        self.assertAlmostEqual(value["u_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()
