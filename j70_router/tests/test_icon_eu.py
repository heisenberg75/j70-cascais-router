from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

from j70_router.wind.icon_eu import download_icon_eu_box, get_latest_icon_eu_run

UTC = timezone.utc


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def post_json(self, endpoint, payload, model=None):
        self.calls.append((endpoint, payload, model))
        return self.rows


class IconEuTests(unittest.TestCase):
    def test_latest_cycle_requires_populated_cascais_wind(self) -> None:
        rows = [
            {"forecasted_at": "2026-09-21T06:00:00Z", "u_ms": 1.0, "v_ms": 2.0},
            {"forecasted_at": "2026-09-21T09:00:00Z", "u_ms": None, "v_ms": None},
        ]
        client = FakeClient(rows)
        run = get_latest_icon_eu_run(client, datetime(2026, 9, 21, 10, tzinfo=UTC))
        self.assertEqual(run, datetime(2026, 9, 21, 6, tzinfo=UTC))
        self.assertEqual(client.calls[0][2], "iconeu")

    def test_interpolation_and_cache_reuse(self) -> None:
        run = datetime(2026, 9, 21, 6, tzinfo=UTC)
        rows = []
        for hour, multiplier in ((16, 1.0), (17, 2.0)):
            for lat in (38.6250, 38.6875):
                for lon in (-9.5625, -9.5000):
                    rows.append(
                        {
                            "forecasted_at": "2026-09-21T06:00:00Z",
                            "forecasted_time": f"2026-09-21T{hour}:00:00Z",
                            "latitude": lat,
                            "longitude": lon,
                            "u_ms": 3.0 * multiplier,
                            "v_ms": 4.0 * multiplier,
                        }
                    )

        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(rows)
            grid = download_icon_eu_box(
                38.6250,
                38.6875,
                -9.5625,
                -9.5000,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                client=client,
                cache_dir=directory,
                forecast_run=run,
            )
            value = grid.wind_at(38.65625, -9.53125, datetime(2026, 9, 21, 16, 30, tzinfo=UTC))
            self.assertAlmostEqual(value["u_ms"], 4.5)
            self.assertAlmostEqual(value["v_ms"], 6.0)
            self.assertAlmostEqual(value["speed_ms"], 7.5)
            self.assertAlmostEqual(value["direction_from_deg"], 216.86989765)
            self.assertTrue(grid.cache_path.exists())
            self.assertEqual(client.calls[0][2], "iconeu")

            second = FakeClient([])
            cached = download_icon_eu_box(
                38.6250,
                38.6875,
                -9.5625,
                -9.5000,
                datetime(2026, 9, 21, 16, tzinfo=UTC),
                datetime(2026, 9, 21, 17, tzinfo=UTC),
                client=second,
                cache_dir=directory,
                forecast_run=run,
            )
            self.assertEqual(second.calls, [])
            self.assertAlmostEqual(
                cached.wind_at(38.65625, -9.53125, datetime(2026, 9, 21, 16, tzinfo=UTC))["speed_ms"],
                5.0,
            )


if __name__ == "__main__":
    unittest.main()
