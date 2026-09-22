from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from j70_router.wind.arome import download_wind_box, get_latest_arome_run

UTC = timezone.utc


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def post_json(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        return self.rows


class AromeTests(unittest.TestCase):
    def test_latest_cycle_comes_from_response(self) -> None:
        rows = [
            {"forecasted_at": "2026-09-21T09:00:00Z", "u_ms": 1.0, "v_ms": 2.0},
            {"forecasted_at": "2026-09-21T12:00:00Z", "u_ms": 2.0, "v_ms": 3.0},
        ]
        run = get_latest_arome_run(FakeClient(rows), datetime(2026, 9, 21, 13, tzinfo=UTC))
        self.assertEqual(run, datetime(2026, 9, 21, 12, tzinfo=UTC))

    def test_latest_cycle_ignores_listed_but_unpopulated_run(self) -> None:
        rows = [
            {"forecasted_at": "2026-09-21T09:00:00Z", "u_ms": 1.0, "v_ms": 2.0},
            {"forecasted_at": "2026-09-21T12:00:00Z", "u_ms": None, "v_ms": None},
        ]
        run = get_latest_arome_run(FakeClient(rows), datetime(2026, 9, 21, 13, tzinfo=UTC))
        self.assertEqual(run, datetime(2026, 9, 21, 9, tzinfo=UTC))

    def test_grid_interpolation_direction_and_cache(self) -> None:
        run = datetime(2026, 9, 21, 12, tzinfo=UTC)
        rows = []
        for hour, multiplier in [(16, 1.0), (17, 2.0)]:
            for lat in (38.60, 38.61):
                for lon in (-9.51, -9.50):
                    rows.append(
                        {
                            "forecasted_at": "2026-09-21T12:00:00Z",
                            "forecasted_time": f"2026-09-21T{hour}:00:00Z",
                            "latitude": lat,
                            "longitude": lon,
                            "u_ms": 3.0 * multiplier,
                            "v_ms": 4.0 * multiplier,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(rows)
            grid = download_wind_box(
                38.60, 38.61, -9.51, -9.50,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                client=client, cache_dir=directory, forecast_run=run,
            )
            value = grid.wind_at(38.605, -9.505, datetime(2026, 9, 21, 16, 30, tzinfo=UTC))
            self.assertAlmostEqual(value["u_ms"], 4.5)
            self.assertAlmostEqual(value["v_ms"], 6.0)
            self.assertAlmostEqual(value["speed_ms"], 7.5)
            self.assertAlmostEqual(value["direction_from_deg"], 216.86989765)
            self.assertTrue(grid.cache_path.exists())
            second = FakeClient([])
            cached = download_wind_box(
                38.60, 38.61, -9.51, -9.50,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                client=second, cache_dir=directory, forecast_run=run,
            )
            self.assertEqual(second.calls, [])
            self.assertAlmostEqual(cached.wind_at(38.605, -9.505, datetime(2026, 9, 21, 16, tzinfo=UTC))["speed_ms"], 5.0)


if __name__ == "__main__":
    unittest.main()
