from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from j70_router.ui.weather_state import (
    WeatherSourceStatus,
    change_forecast_time,
    first_valid_weather_source,
    mark_update_invalidates_route,
    validate_weather_dataset,
)


UTC = timezone.utc
TIME = datetime(2026, 9, 21, 16, tzinfo=UTC)


def dataset(u: float = 3.0, v: float = 4.0):
    grid = SimpleNamespace(
        latitudes=np.array([38.5, 38.8]),
        longitudes=np.array([-9.8, -9.2]),
        times=(datetime(2026, 9, 21, 15, tzinfo=UTC), datetime(2026, 9, 21, 17, tzinfo=UTC)),
        u_mps=np.full((2, 2, 2), u),
        v_mps=np.full((2, 2, 2), v),
    )
    return SimpleNamespace(grid=grid)


class WeatherStateTests(unittest.TestCase):
    def test_null_uv_is_not_available_for_cascais(self) -> None:
        result = validate_weather_dataset(dataset(float("nan"), float("nan")), (38.6, 38.7, -9.6, -9.4), TIME)
        self.assertEqual(result.status, WeatherSourceStatus.NOT_AVAILABLE_FOR_LOCATION)

    def test_finite_icon_style_grid_is_available(self) -> None:
        result = validate_weather_dataset(dataset(), (38.6, 38.7, -9.6, -9.4), TIME)
        self.assertEqual(result.status, WeatherSourceStatus.AVAILABLE)

    def test_invalid_arome_candidate_falls_through_to_valid_icon(self) -> None:
        invalid_arome = dataset(float("nan"), float("nan"))
        valid_icon = dataset()
        name, selected = first_valid_weather_source(
            (("arome_france", invalid_arome), ("icon_eu", valid_icon)),
            (38.6, 38.7, -9.6, -9.4),
            TIME,
        )
        self.assertEqual(name, "icon_eu")
        self.assertIs(selected, valid_icon)

    def test_forecast_time_change_keeps_loaded_dataset(self) -> None:
        weather = dataset()
        later = datetime(2026, 9, 21, 17, tzinfo=UTC)
        state: dict[str, object] = {
            "selected_forecast_time": TIME,
            "forecast_dataset": weather,
            "weather_source": "icon_eu",
            "last_route": object(),
            "last_route_signature": ("old",),
        }
        self.assertTrue(change_forecast_time(state, later))
        self.assertIs(state["forecast_dataset"], weather)
        self.assertEqual(state["weather_source"], "icon_eu")
        self.assertIsNone(state["last_route"])

    def test_moving_mark_invalidates_route_without_clearing_weather(self) -> None:
        weather = dataset()
        state: dict[str, object] = {
            "leeward_lat": 38.63,
            "leeward_lon": -9.49,
            "last_route": object(),
            "last_route_signature": ("old",),
            "forecast_dataset": weather,
            "weather_source": "icon_eu",
        }
        self.assertTrue(mark_update_invalidates_route(state, "leeward", 38.631, -9.491))
        self.assertIsNone(state["last_route"])
        self.assertIs(state["forecast_dataset"], weather)
        self.assertEqual(state["weather_source"], "icon_eu")
        self.assertFalse(mark_update_invalidates_route(state, "leeward", 38.631, -9.491))


if __name__ == "__main__":
    unittest.main()
