from __future__ import annotations

import unittest
from datetime import datetime, timezone

from j70_router.sailing.geometry import LocalCartesian
from j70_router.routing.optimizer import RoutingConfig, route_race
from j70_router.sailing.polar import ConstantSpeedPolar
from j70_router.ui.map_view import (
    _airflow_endpoint,
    _wind_rgba,
    apply_mark_click,
    build_course_map,
    mark_prefix_for_event,
    process_map_interaction,
    update_view_state,
)
from j70_router.wind.synthetic import SyntheticWindField


class MapViewTests(unittest.TestCase):
    def test_drag_event_uses_marker_identity_without_changing_selection(self) -> None:
        self.assertEqual(
            mark_prefix_for_event("Windward mark", "Leeward / start mark"),
            "leeward",
        )
        self.assertEqual(mark_prefix_for_event("Windward mark", None), "windward")

    def test_wind_raster_uses_fixed_scale_and_alpha(self) -> None:
        import numpy as np

        rgba = _wind_rgba(np.array([[-1.0, 10.0, 25.0]]), alpha=160)
        self.assertEqual(rgba.shape, (1, 3, 4))
        self.assertTrue((rgba[..., 3] == 160).all())
        self.assertFalse((rgba[0, 0, :3] == rgba[0, 2, :3]).all())

    def test_component_callback_persists_view_before_rerun(self) -> None:
        route = object()
        weather = object()
        state = {
            "map_center": (38.64, -9.50),
            "map_zoom": 12,
            "selected_mark": "Windward mark",
            "last_processed_click": None,
            "windward_lat": 38.65,
            "windward_lon": -9.51,
            "last_route": route,
            "last_route_signature": ("old",),
            "forecast_dataset": weather,
        }
        viewport_changed, mark_changed = process_map_interaction(
            state,
            {
                "center": {"lat": 38.67, "lng": -9.57},
                "zoom": 15,
                "last_clicked": {"lat": 38.66, "lng": -9.56},
                "last_object_clicked_tooltip": "Windward mark",
            },
        )
        self.assertTrue(viewport_changed)
        self.assertTrue(mark_changed)
        self.assertEqual(state["map_center"], (38.67, -9.57))
        self.assertEqual(state["map_zoom"], 15)
        self.assertEqual(state["windward_lat"], 38.66)
        self.assertEqual(state["windward_lon"], -9.56)
        self.assertIsNone(state["last_route"])
        self.assertIs(state["forecast_dataset"], weather)

    def test_viewport_state_changes_only_for_material_pan_or_zoom(self) -> None:
        state = {"map_center": (38.64, -9.50), "map_zoom": 12}
        self.assertFalse(update_view_state(state, (38.640000001, -9.500000001), 12))
        self.assertTrue(update_view_state(state, (38.66, -9.55), 15))
        self.assertEqual(state["map_center"], (38.66, -9.55))
        self.assertEqual(state["map_zoom"], 15)

    def test_click_updates_only_selected_authoritative_mark(self) -> None:
        state = {
            "leeward_lat": 38.6, "leeward_lon": -9.5,
            "windward_lat": 38.7, "windward_lon": -9.4,
        }
        apply_mark_click(state, "Leeward / start mark", 38.61234567, -9.51234567)
        self.assertEqual(state["leeward_lat"], 38.612346)
        self.assertEqual(state["leeward_lon"], -9.512346)
        self.assertEqual(state["windward_lat"], 38.7)
        apply_mark_click(state, "Windward mark", 38.6234567, -9.5234567)
        self.assertEqual(state["windward_lat"], 38.623457)
        self.assertEqual(state["windward_lon"], -9.523457)

    def test_airflow_arrow_uses_uv_toward_direction(self) -> None:
        north = _airflow_endpoint(38.6, -9.5, 0.0, 5.0, 500.0)
        west = _airflow_endpoint(38.6, -9.5, -5.0, 0.0, 500.0)
        self.assertGreater(north[0], 38.6)
        self.assertAlmostEqual(north[1], -9.5, places=6)
        self.assertLess(west[1], -9.5)
        self.assertAlmostEqual(west[0], 38.6, places=6)

    def test_map_contains_both_mark_labels_and_course(self) -> None:
        frame = LocalCartesian(38.64, -9.50)
        leeward = frame.to_xy(38.635, -9.495)
        windward = frame.to_xy(38.645, -9.505)
        html = build_course_map(
            (38.635, -9.495), (38.645, -9.505), frame, leeward, windward,
            SyntheticWindField.uniform(5.0, 315.0),
            datetime(2026, 9, 21, 16, tzinfo=timezone.utc),
        ).get_root().render()
        self.assertIn("Leeward / start mark", html)
        self.assertIn("Windward mark", html)
        self.assertIn("Course axis", html)
        self.assertIn("draggable", html)
        self.assertIn("openstreetmap", html.lower())
        self.assertIn("Wind speed [kt]", html)

    def test_route_is_rendered_as_port_and_starboard_segments(self) -> None:
        frame = LocalCartesian(38.64, -9.50)
        leeward = (0.0, 0.0)
        windward = (0.0, 500.0)
        start = datetime(2026, 9, 21, 16, tzinfo=timezone.utc)
        wind = SyntheticWindField.uniform(5.0, 0.0)
        race = route_race(
            leeward, windward, start, wind, ConstantSpeedPolar(),
            max_tacks=2, max_gybes=2,
            config=RoutingConfig(
                dt_seconds=10, spatial_bin_m=25, mark_radius_m=15,
                beam_width=3000, max_leg_time_seconds=1800,
            ),
        )
        html = build_course_map(
            frame.to_latlon(*leeward), frame.to_latlon(*windward), frame,
            leeward, windward, wind, start, race=race,
        ).get_root().render()
        self.assertIn("Optimized J/70 route", html)
        self.assertIn("#2474b5", html)
        self.assertIn("#e67e22", html)


if __name__ == "__main__":
    unittest.main()
