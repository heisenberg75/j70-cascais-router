from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from j70_router.routing.simulator import TimeClampedCurrentField
from j70_router.ui.time_display import format_cascais_time
from j70_router.wind.base import TimeClampedWindField, WindSample


UTC = timezone.utc


class _RecordingWind:
    def __init__(self) -> None:
        self.queries: list[datetime] = []

    def wind_at(self, position: tuple[float, float], time: datetime) -> WindSample:
        self.queries.append(time)
        return WindSample(float(time.hour), 0.0)


class _RecordingCurrent:
    def __init__(self) -> None:
        self.queries: list[datetime] = []

    def current_at(self, position: tuple[float, float], time: datetime) -> tuple[float, float]:
        self.queries.append(time)
        return float(time.hour), 0.0


class TimeHandlingTests(unittest.TestCase):
    def test_cascais_display_applies_daylight_saving_time(self) -> None:
        winter = datetime(2026, 1, 15, 16, tzinfo=UTC)
        summer = datetime(2026, 9, 22, 16, tzinfo=UTC)
        self.assertEqual(format_cascais_time(winter), "2026-01-15 16:00 WET")
        self.assertEqual(format_cascais_time(summer), "2026-09-22 17:00 WEST")

    def test_naive_display_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            format_cascais_time(datetime(2026, 9, 22, 16))

    def test_wind_holds_last_frame_after_forecast_horizon(self) -> None:
        first = datetime(2026, 9, 22, 16, tzinfo=UTC)
        last = first + timedelta(hours=2)
        source = _RecordingWind()
        field = TimeClampedWindField(source, first, last)
        sample = field.wind_at((0.0, 0.0), last + timedelta(hours=4))
        self.assertEqual(source.queries, [last])
        self.assertEqual(sample.u_mps, 18.0)

    def test_current_holds_last_frame_after_forecast_horizon(self) -> None:
        first = datetime(2026, 9, 22, 16, tzinfo=UTC)
        last = first + timedelta(hours=2)
        source = _RecordingCurrent()
        field = TimeClampedCurrentField(source, first, last)
        value = field.current_at((0.0, 0.0), last + timedelta(hours=4))
        self.assertEqual(source.queries, [last])
        self.assertEqual(value, (18.0, 0.0))


if __name__ == "__main__":
    unittest.main()
