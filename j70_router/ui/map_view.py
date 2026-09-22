"""Folium layers for marks, native-grid wind, and routed tracks."""

from __future__ import annotations

from datetime import datetime
from bisect import bisect_right
from math import ceil
from typing import MutableMapping

import folium
import numpy as np
from branca.element import Element, MacroElement
from folium.plugins import PolyLineTextPath
from jinja2 import Template

from ..current.copernicus import CopernicusCurrentGrid
from ..routing.optimizer import MultiLapResult, RaceResult
from ..sailing.geometry import LocalCartesian
from ..sailing.polar import KNOT_TO_MPS
from ..wind.arome import AromeWindGrid
from ..wind.icon_eu import IconEuWindGrid
from ..wind.base import WindField
from .weather_state import mark_update_invalidates_route

PORT_COLOUR = "#2474b5"
STARBOARD_COLOUR = "#e67e22"
WIND_COLOURS = (
    "#5b47b7", "#4169c1", "#348fc2", "#2fb7b0", "#47c875",
    "#9bd44e", "#e5d43c", "#f6a035", "#ef642f", "#b92f45",
)
CURRENT_COLOURS = (
    "#29235c", "#31499b", "#287fbd", "#1eb6bd", "#42ca85",
    "#9bd44e", "#e8d33e", "#f39b32", "#dc4a35",
)


def _wind_colour(speed_knots: float) -> str:
    """Return a stable discrete color on the fixed 0–20 kt display scale."""

    fraction = float(np.clip(speed_knots, 0.0, 20.0)) / 20.0
    return WIND_COLOURS[min(int(fraction * len(WIND_COLOURS)), len(WIND_COLOURS) - 1)]


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _speed_rgba(
    values_knots: np.ndarray,
    maximum_knots: float,
    colours: tuple[str, ...],
    alpha: int = 255,
) -> np.ndarray:
    values = np.asarray(values_knots, dtype=float)
    valid = np.isfinite(values)
    clipped = np.clip(np.where(valid, values, 0.0), 0.0, maximum_knots)
    positions = np.linspace(0.0, maximum_knots, len(colours))
    palette = np.asarray([_hex_rgb(value) for value in colours], dtype=float)
    rgba = np.empty(values.shape + (4,), dtype=np.uint8)
    for channel in range(3):
        rgba[..., channel] = np.interp(clipped, positions, palette[:, channel]).astype(np.uint8)
    rgba[..., 3] = np.where(valid, alpha, 0).astype(np.uint8)
    return rgba


def _wind_rgba(values_knots: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Map wind speed to a continuous fixed 0–20 kt RGBA scale."""

    return _speed_rgba(values_knots, 20.0, WIND_COLOURS, alpha)


def _grid_uv_at_time(
    source: AromeWindGrid | IconEuWindGrid | CopernicusCurrentGrid, valid_time: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the native U/V arrays once for efficient map rendering."""

    times = source.grid.times
    if valid_time < times[0] or valid_time > times[-1]:
        raise ValueError("forecast valid time is outside the loaded dataset")
    upper = bisect_right(times, valid_time)
    if upper == 0:
        return source.grid.u_mps[0], source.grid.v_mps[0]
    if upper >= len(times):
        return source.grid.u_mps[-1], source.grid.v_mps[-1]
    lower = upper - 1
    if times[lower] == valid_time:
        return source.grid.u_mps[lower], source.grid.v_mps[lower]
    fraction = (valid_time - times[lower]).total_seconds() / (times[upper] - times[lower]).total_seconds()
    u = source.grid.u_mps[lower] * (1.0 - fraction) + source.grid.u_mps[upper] * fraction
    v = source.grid.v_mps[lower] * (1.0 - fraction) + source.grid.v_mps[upper] * fraction
    return u, v


def _add_wind_legend(map_object: folium.Map) -> None:
    gradient = ", ".join(WIND_COLOURS)
    map_object.get_root().html.add_child(
        Element(
            f"""<div style="position:fixed;bottom:24px;right:12px;z-index:9999;
            background:white;padding:7px 9px;border:1px solid #777;border-radius:3px;
            font:12px sans-serif;box-shadow:0 1px 4px #999">
            <div style="font-weight:600;margin-bottom:4px">Wind speed [kt]</div>
            <div style="width:180px;height:12px;background:linear-gradient(to right,{gradient})"></div>
            <div style="display:flex;justify-content:space-between">
              <span>0</span><span>5</span><span>10</span><span>15</span><span>20+</span>
            </div>
            </div>"""
        )
    )


def _add_current_legend(map_object: folium.Map) -> None:
    gradient = ", ".join(CURRENT_COLOURS)
    map_object.get_root().html.add_child(
        Element(
            f"""<div style="position:fixed;bottom:24px;right:12px;z-index:9999;
            background:white;padding:7px 9px;border:1px solid #777;border-radius:3px;
            font:12px sans-serif;box-shadow:0 1px 4px #999">
            <div style="font-weight:600;margin-bottom:4px">Surface current [kt]</div>
            <div style="width:180px;height:12px;background:linear-gradient(to right,{gradient})"></div>
            <div style="display:flex;justify-content:space-between">
              <span>0</span><span>0.5</span><span>1</span><span>1.5</span><span>2+</span>
            </div>
            </div>"""
        )
    )


def update_view_state(
    state: MutableMapping[str, object],
    center: tuple[float, float] | None,
    zoom: int | None,
    tolerance_deg: float = 1e-8,
) -> bool:
    """Persist a returned Leaflet viewport without creating write/rerun noise."""

    changed = False
    if center is not None:
        previous = tuple(state["map_center"])
        if any(abs(float(a) - float(b)) > tolerance_deg for a, b in zip(center, previous)):
            state["map_center"] = (float(center[0]), float(center[1]))
            changed = True
    if zoom is not None and int(zoom) != int(state["map_zoom"]):
        state["map_zoom"] = int(zoom)
        changed = True
    return changed


class _DragBridge(MacroElement):
    """Forward Leaflet marker drag positions through streamlit-folium clicks."""

    _template = Template(
        """{% macro script(this, kwargs) %}
        (function() {
          function enableJ70Drag() {
            var markers = [
              [{{ this.leeward_marker }}, 'Leeward / start mark'],
              [{{ this.windward_marker }}, 'Windward mark']
            ];
            markers.forEach(function(item) {
              var marker = item[0];
              if (!marker || !marker.dragging || marker._j70DragBridge) return;
              marker._j70DragBridge = true;
              marker.dragging.enable();
              marker.on('dragend', function(event) {
                marker.fire('click');
                {{ this.map_name }}.fire('click', {latlng: event.target.getLatLng()});
              });
            });
          }
          enableJ70Drag();
        })();
        {% endmacro %}"""
    )

    def __init__(self, map_name: str, leeward_marker: str, windward_marker: str) -> None:
        super().__init__()
        self._name = "j70_drag_bridge"
        self.map_name = map_name
        self.leeward_marker = leeward_marker
        self.windward_marker = windward_marker


def apply_mark_click(
    state: MutableMapping[str, object], selected_mark: str, lat: float, lon: float
) -> None:
    """Apply one map selection to the authoritative six-decimal coordinates."""

    prefix = "leeward" if selected_mark == "Leeward / start mark" else "windward"
    state[f"{prefix}_lat"] = round(float(lat), 6)
    state[f"{prefix}_lon"] = round(float(lon), 6)


def mark_prefix_for_event(selected_mark: str, marker_tooltip: str | None) -> str:
    """Resolve a map event without mutating an already-rendered Streamlit widget."""

    event_mark = marker_tooltip if marker_tooltip in ("Leeward / start mark", "Windward mark") else selected_mark
    return "leeward" if event_mark == "Leeward / start mark" else "windward"


def process_map_interaction(
    state: MutableMapping[str, object], map_data: dict[str, object] | None
) -> tuple[bool, bool]:
    """Persist viewport and drag/click state before Streamlit reruns the script."""

    if not map_data:
        return False, False
    raw_center = map_data.get("center")
    center = None
    if isinstance(raw_center, dict) and "lat" in raw_center and "lng" in raw_center:
        center = (float(raw_center["lat"]), float(raw_center["lng"]))
    raw_zoom = map_data.get("zoom")
    viewport_changed = update_view_state(
        state, center, int(raw_zoom) if raw_zoom is not None else None
    )

    click = map_data.get("last_clicked")
    if not isinstance(click, dict) or "lat" not in click or "lng" not in click:
        return viewport_changed, False
    tooltip = map_data.get("last_object_clicked_tooltip")
    prefix = mark_prefix_for_event(
        str(state["selected_mark"]), str(tooltip) if tooltip is not None else None
    )
    lat, lon = float(click["lat"]), float(click["lng"])
    click_key = (prefix, round(lat, 7), round(lon, 7))
    if click_key == state.get("last_processed_click"):
        return viewport_changed, False
    state["last_processed_click"] = click_key
    return viewport_changed, mark_update_invalidates_route(state, prefix, lat, lon)


def _mark_icon(letter: str, colour: str) -> folium.DivIcon:
    return folium.DivIcon(
        icon_size=(30, 30),
        icon_anchor=(15, 15),
        html=(
            f'<div style="width:30px;height:30px;border-radius:50%;background:{colour};'
            'border:2px solid white;color:white;font-weight:700;font-size:16px;'
            f'line-height:26px;text-align:center;box-shadow:0 1px 4px #333">{letter}</div>'
        ),
    )


def _airflow_endpoint(lat: float, lon: float, u_mps: float, v_mps: float, length_m: float) -> tuple[float, float]:
    speed = float(np.hypot(u_mps, v_mps))
    if speed < 1e-9:
        return lat, lon
    frame = LocalCartesian(lat, lon)
    return frame.to_latlon(u_mps / speed * length_m, v_mps / speed * length_m)


def _add_arrow(
    layer: folium.FeatureGroup,
    lat: float,
    lon: float,
    u_mps: float,
    v_mps: float,
    length_m: float = 650.0,
) -> None:
    end_lat, end_lon = _airflow_endpoint(lat, lon, u_mps, v_mps, length_m)
    line = folium.PolyLine(
        [(lat, lon), (end_lat, end_lon)], color="#17324d", weight=1.6, opacity=0.85
    ).add_to(layer)
    PolyLineTextPath(
        line,
        "➤",
        repeat=False,
        center=True,
        offset=5,
        attributes={"fill": "#17324d", "font-size": "16", "font-weight": "bold"},
    ).add_to(layer)


def _add_gridded_wind(
    map_object: folium.Map,
    source: AromeWindGrid | IconEuWindGrid,
    valid_time: datetime,
) -> None:
    lats = source.grid.latitudes
    lons = source.grid.longitudes
    u, v = _grid_uv_at_time(source, valid_time)
    values = np.hypot(u, v) / KNOT_TO_MPS
    lat_step = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.01
    lon_step = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.01
    folium.raster_layers.ImageOverlay(
        image=_wind_rgba(values),
        bounds=[
            (float(lats[0] - lat_step / 2), float(lons[0] - lon_step / 2)),
            (float(lats[-1] + lat_step / 2), float(lons[-1] + lon_step / 2)),
        ],
        origin="lower",
        pixelated=False,
        opacity=0.84,
        name=f"{source.model_name} wind speed · native {source.grid_resolution_deg:g}°",
        show=True,
    ).add_to(map_object)
    _add_wind_legend(map_object)

    arrows = folium.FeatureGroup(name="Wind arrows · air movement direction", show=True)
    # Use native grid points, thinned adaptively to one or two cells while
    # keeping a large 10 km-margin request readable.
    stride = min(2, max(1, ceil(max(len(lats), len(lons)) / 16)))
    for lat in lats[::stride]:
        for lon in lons[::stride]:
            value = source.wind_at(float(lat), float(lon), valid_time)
            _add_arrow(arrows, float(lat), float(lon), value["u_ms"], value["v_ms"])
    arrows.add_to(map_object)


def _add_gridded_current(
    map_object: folium.Map,
    source: CopernicusCurrentGrid,
    valid_time: datetime,
) -> None:
    lats = source.grid.latitudes
    lons = source.grid.longitudes
    u, v = _grid_uv_at_time(source, valid_time)
    values = np.hypot(u, v) / KNOT_TO_MPS
    lat_step = float(np.median(np.diff(lats))) if len(lats) > 1 else source.grid_resolution_deg
    lon_step = float(np.median(np.diff(lons))) if len(lons) > 1 else source.grid_resolution_deg
    folium.raster_layers.ImageOverlay(
        image=_speed_rgba(values, 2.0, CURRENT_COLOURS),
        bounds=[
            (float(lats[0] - lat_step / 2), float(lons[0] - lon_step / 2)),
            (float(lats[-1] + lat_step / 2), float(lons[-1] + lon_step / 2)),
        ],
        origin="lower",
        pixelated=False,
        opacity=0.84,
        name=f"Copernicus IBI current · native {source.grid_resolution_deg:g}°",
        show=True,
    ).add_to(map_object)
    _add_current_legend(map_object)

    arrows = folium.FeatureGroup(name="Current arrows · water movement direction", show=True)
    stride = max(1, ceil(max(len(lats), len(lons)) / 16))
    for yi in range(0, len(lats), stride):
        for xi in range(0, len(lons), stride):
            if not np.isfinite(u[yi, xi]) or not np.isfinite(v[yi, xi]):
                continue
            _add_arrow(
                arrows, float(lats[yi]), float(lons[xi]), float(u[yi, xi]), float(v[yi, xi]),
                length_m=500.0,
            )
    arrows.add_to(map_object)


def _add_synthetic_wind(
    map_object: folium.Map,
    wind: WindField,
    frame: LocalCartesian,
    leeward: tuple[float, float],
    windward: tuple[float, float],
    valid_time: datetime,
) -> None:
    layer = folium.FeatureGroup(name="Synthetic wind arrows", show=True)
    xs = np.linspace(min(leeward[0], windward[0]) - 5000, max(leeward[0], windward[0]) + 5000, 17)
    ys = np.linspace(min(leeward[1], windward[1]) - 5000, max(leeward[1], windward[1]) + 5000, 17)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1000.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1000.0
    speeds = np.empty((len(ys), len(xs)), dtype=float)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            sample = wind.wind_at((float(x), float(y)), valid_time)
            speeds[yi, xi] = sample.speed_mps / KNOT_TO_MPS
            if yi % 4 or xi % 4:
                continue
            lat, lon = frame.to_latlon(float(x), float(y))
            _add_arrow(layer, lat, lon, sample.u_mps, sample.v_mps)
    south_west = frame.to_latlon(float(xs[0] - dx / 2), float(ys[0] - dy / 2))
    north_east = frame.to_latlon(float(xs[-1] + dx / 2), float(ys[-1] + dy / 2))
    folium.raster_layers.ImageOverlay(
        image=_wind_rgba(speeds),
        bounds=[south_west, north_east],
        origin="lower",
        pixelated=False,
        opacity=0.82,
        name="Synthetic wind speed · display interpolation",
        show=True,
    ).add_to(map_object)
    _add_wind_legend(map_object)
    layer.add_to(map_object)


def _add_route(
    map_object: folium.Map,
    race: RaceResult | MultiLapResult,
    frame: LocalCartesian,
) -> None:
    layer = folium.FeatureGroup(name="Optimized J/70 route", show=True)
    legs = race.legs if isinstance(race, MultiLapResult) else (race.upwind, race.downwind)
    for leg in legs:
        for previous, current in zip(leg.points, leg.points[1:]):
            colour = PORT_COLOUR if current.side == "port" else STARBOARD_COLOUR
            folium.PolyLine(
                [frame.to_latlon(previous.x, previous.y), frame.to_latlon(current.x, current.y)],
                color=colour,
                weight=5,
                opacity=0.95,
                tooltip=f"{leg.mode} · {current.side}",
            ).add_to(layer)
    for leg in legs:
        for maneuver in leg.maneuvers:
            if leg.mode == "upwind":
                marker = folium.CircleMarker(
                    frame.to_latlon(maneuver.x, maneuver.y), radius=5, color="white", weight=1,
                    fill=True, fill_color="black", fill_opacity=1,
                    tooltip=f"Tack · {maneuver.time:%H:%M:%S} UTC",
                )
            else:
                marker = folium.RegularPolygonMarker(
                    frame.to_latlon(maneuver.x, maneuver.y), number_of_sides=4, radius=6,
                    color="white", weight=1, fill=True, fill_color="black", fill_opacity=1,
                    tooltip=f"Gybe · {maneuver.time:%H:%M:%S} UTC",
                )
            marker.add_to(layer)
    layer.add_to(map_object)


def build_course_map(
    leeward_latlon: tuple[float, float],
    windward_latlon: tuple[float, float],
    frame: LocalCartesian,
    leeward_xy: tuple[float, float],
    windward_xy: tuple[float, float],
    wind: WindField,
    valid_time: datetime,
    arome_source: AromeWindGrid | IconEuWindGrid | None = None,
    race: RaceResult | MultiLapResult | None = None,
    centre: tuple[float, float] | None = None,
    zoom: int = 12,
    show_wind: bool = True,
    current_source: CopernicusCurrentGrid | None = None,
    current_time: datetime | None = None,
) -> folium.Map:
    """Build the interactive map without mutating application state."""

    location = centre or (
        (leeward_latlon[0] + windward_latlon[0]) / 2,
        (leeward_latlon[1] + windward_latlon[1]) / 2,
    )
    # OpenStreetMap requires no private token. The course overlays remain useful
    # if tiles are unavailable because all tactical layers are local Folium data.
    map_object = folium.Map(location=location, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    if show_wind:
        if arome_source is not None:
            _add_gridded_wind(map_object, arome_source, valid_time)
        else:
            _add_synthetic_wind(map_object, wind, frame, leeward_xy, windward_xy, valid_time)
    elif current_source is not None and current_time is not None:
        _add_gridded_current(map_object, current_source, current_time)

    folium.PolyLine(
        [leeward_latlon, windward_latlon],
        color="#333333",
        weight=2,
        dash_array="8 7",
        opacity=0.9,
        tooltip="Course axis",
    ).add_to(map_object)
    if race is not None:
        _add_route(map_object, race, frame)
    leeward_marker = folium.Marker(
        leeward_latlon, icon=_mark_icon("L", "#198754"), tooltip="Leeward / start mark",
        draggable=True,
    ).add_to(map_object)
    windward_marker = folium.Marker(
        windward_latlon, icon=_mark_icon("W", "#c62828"), tooltip="Windward mark",
        draggable=True,
    ).add_to(map_object)
    # streamlit-folium reports marker clicks but not dragend directly. A drag
    # emits the same Leaflet click payload, with the marker tooltip identifying
    # which authoritative coordinate should be updated by the app.
    map_object.add_child(
        _DragBridge(map_object.get_name(), leeward_marker.get_name(), windward_marker.get_name())
    )
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object
