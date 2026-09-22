"""Local Streamlit engineering UI for the Cascais J/70 router."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from math import atan2, degrees

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from j70_router.current.copernicus import (
    credentials_configured as copernicus_credentials_configured,
    download_current_box,
    load_cached_current,
)
from j70_router.plotting import plot_wind_field_map
from j70_router.routing.optimizer import (
    RouteNotFoundError,
    RoutingConfig,
    adaptive_routing_config,
    heading_sector_margin_deg,
    route_laps,
)
from j70_router.routing.simulator import TimeClampedCurrentField
from j70_router.sailing.geometry import LocalCartesian, distance
from j70_router.sailing.j70_polar import J70Polar, J70SpeedModel
from j70_router.sailing.polar import KNOT_TO_MPS
from j70_router.sailing.targets import TargetAngleModel, expected_vmg_knots, target_twa
from j70_router.ui.map_view import build_course_map, process_map_interaction
from j70_router.ui.time_display import format_cascais_time
from j70_router.ui.weather_state import WeatherSourceStatus, validate_weather_dataset
from j70_router.wind.base import TimeClampedWindField, WindOutOfBoundsError
from j70_router.wind.icon_eu import download_icon_eu_box, get_latest_icon_eu_run, load_cached_icon_eu
from j70_router.wind.ipma_portugal import (
    download_ipma_portugal_box,
    get_latest_ipma_portugal_run,
    load_cached_ipma_portugal,
)
from j70_router.wind.synthetic import SyntheticWindField

UTC = timezone.utc
# Appendix B places Area 5 at 2.0 NM on bearing 220 degrees from Clube
# Naval de Cascais. These editable marks straddle that approximate centre.
AREA_5_CENTER = (38.6692624266, -9.4463238667)
DEFAULT_LEEWARD = (38.6647658285, -9.4463238667)
DEFAULT_WINDWARD = (38.6737590248, -9.4463238667)
MARK_OPTIONS = ("Leeward / start mark", "Windward mark")
POLAR_DATA_VERSION = "orc-2022-vpp-1.04"


@st.cache_resource
def get_polar(data_version: str) -> J70Polar:
    del data_version  # Included in the cache key so table revisions reload.
    return J70Polar()


def utc_hour() -> datetime:
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def default_race_time() -> datetime:
    """Development default requested for the current Cascais test race."""

    today = datetime.now(UTC)
    return today.replace(hour=16, minute=0, second=0, microsecond=0)


def initialize_state() -> None:
    defaults = {
        "leeward_lat": DEFAULT_LEEWARD[0],
        "leeward_lon": DEFAULT_LEEWARD[1],
        "windward_lat": DEFAULT_WINDWARD[0],
        "windward_lon": DEFAULT_WINDWARD[1],
        "selected_mark": MARK_OPTIONS[0],
        "loaded_weather_run": None,
        "forecast_dataset": None,
        "selected_forecast_time": default_race_time(),
        "weather_source": "synthetic",
        "requested_weather_source": "icon_eu",
        "last_route": None,
        "last_route_signature": None,
        "last_route_execution_seconds": None,
        "last_route_error": None,
        "weather_message": "Using synthetic/test wind.",
        "current_dataset": None,
        "current_message": "Copernicus Marine current has not been loaded.",
        "crew_speed_factor": 0.93,
        "heading_sigma_deg": 3,
        "maximum_tacks": 4,
        "maximum_gybes": 4,
        "minimum_tacks": 0,
        "minimum_gybes": 0,
        "tack_penalty_seconds": 7.0,
        "gybe_penalty_seconds": 5.0,
        "routing_model_label": "Robust",
        "routing_timestep_seconds": 10.0,
        "spatial_resolution_m": 30.0,
        "maneuver_separation_seconds": 20.0,
        "last_processed_click": None,
        "map_center": ((DEFAULT_LEEWARD[0] + DEFAULT_WINDWARD[0]) / 2, (DEFAULT_LEEWARD[1] + DEFAULT_WINDWARD[1]) / 2),
        "map_zoom": 12,
        "map_field": "Wind",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Keep an existing browser session working after the state names changed.
    if st.session_state.weather_source == "synthetic" and st.session_state.get("weather_mode") not in (None, "synthetic"):
        st.session_state.weather_source = st.session_state.weather_mode
    if st.session_state.loaded_weather_run is None and st.session_state.get("loaded_arome_run") is not None:
        st.session_state.loaded_weather_run = st.session_state.loaded_arome_run
    if st.session_state.weather_source not in ("icon_eu", "ipma_portugal", "synthetic"):
        st.session_state.weather_source = "synthetic"
        st.session_state.forecast_dataset = None
        st.session_state.loaded_weather_run = None
        st.session_state.weather_message = "AROME-France is not available for Cascais; choose ICON-EU or Synthetic."


def geometry() -> tuple[
    LocalCartesian, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
]:
    leeward = (float(st.session_state.leeward_lat), float(st.session_state.leeward_lon))
    windward = (float(st.session_state.windward_lat), float(st.session_state.windward_lon))
    frame = LocalCartesian((leeward[0] + windward[0]) / 2, (leeward[1] + windward[1]) / 2)
    return frame, frame.to_xy(*leeward), frame.to_xy(*windward), leeward, windward


def course_bearing(start: tuple[float, float], target: tuple[float, float]) -> float:
    return degrees(atan2(target[0] - start[0], target[1] - start[1])) % 360.0


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes:d}m {remainder:02d}s"


def available_forecast_times() -> tuple[datetime, ...]:
    source = st.session_state.forecast_dataset
    if st.session_state.weather_source != "synthetic" and source is not None:
        return source.grid.times
    first = utc_hour()
    return tuple(first + timedelta(hours=hour) for hour in range(24))


def select_forecast_time(value: datetime) -> None:
    st.session_state.selected_forecast_time = value
    st.session_state.forecast_time_widget = value


def select_nearest_forecast_time(times: tuple[datetime, ...]) -> None:
    """Keep the requested race hour when a newly loaded run contains it."""

    requested = st.session_state.selected_forecast_time
    # A weather action happens after the top selector has been instantiated.
    # Update only the authoritative value here; the explicit rerun then rebuilds
    # the widget from the new dataset without mutating an active widget key.
    st.session_state.selected_forecast_time = min(
        times, key=lambda value: abs((value - requested).total_seconds())
    )


def invalidate_route() -> None:
    st.session_state.last_route = None
    st.session_state.last_route_signature = None
    st.session_state.last_route_error = None


def sampled_course_reachability(
    start: tuple[float, float],
    target: tuple[float, float],
    start_time: datetime,
    wind,
    target_model: TargetAngleModel,
    mode: str,
    sample_span_seconds: float,
) -> tuple[bool, float]:
    """Check representative forecast samples against the legal-heading cone."""

    margins: list[float] = []
    for fraction in (0.0, 0.5, 1.0):
        position = (
            start[0] + (target[0] - start[0]) * fraction,
            start[1] + (target[1] - start[1]) * fraction,
        )
        try:
            sample = wind.wind_at(
                position, start_time + timedelta(seconds=sample_span_seconds * fraction)
            )
        except WindOutOfBoundsError:
            continue
        margins.append(
            heading_sector_margin_deg(
                start,
                target,
                sample.direction_deg,
                target_model(sample.speed_mps, mode),
                mode,
            )
        )
    return (not margins or max(margins) >= 0.0), (max(margins) if margins else float("nan"))


def recenter_on_marks() -> None:
    invalidate_route()


def download_bounds(
    frame: LocalCartesian,
    leeward_xy: tuple[float, float],
    windward_xy: tuple[float, float],
) -> tuple[float, float, float, float]:
    margin_m = 10_000.0
    min_x, max_x = sorted((leeward_xy[0], windward_xy[0]))
    min_y, max_y = sorted((leeward_xy[1], windward_xy[1]))
    south, west = frame.to_latlon(min_x - margin_m, min_y - margin_m)
    north, east = frame.to_latlon(max_x + margin_m, max_y + margin_m)
    return south, north, west, east


def make_route_signature(
    selected_time: datetime,
    crew_factor: float,
    heading_sigma: int,
    max_tacks: int,
    max_gybes: int,
    min_tacks: int,
    min_gybes: int,
    tack_penalty_seconds: float,
    gybe_penalty_seconds: float,
    maneuver_separation_seconds: float,
    routing_model: str,
    timestep: float,
    spatial_resolution: float,
) -> tuple:
    run = st.session_state.loaded_weather_run
    source_id = f"{st.session_state.weather_source}:{run.isoformat()}" if run else "synthetic"
    current_dataset = st.session_state.get("current_dataset")
    current_id = (
        f"{current_dataset.cache_path}:{current_dataset.grid.times[0].isoformat()}"
        if current_dataset is not None else "no-current"
    )
    return (
        "two-laps-si-2026",
        round(float(st.session_state.leeward_lat), 7),
        round(float(st.session_state.leeward_lon), 7),
        round(float(st.session_state.windward_lat), 7),
        round(float(st.session_state.windward_lon), 7),
        selected_time.isoformat(),
        float(crew_factor),
        int(heading_sigma),
        int(max_tacks),
        int(max_gybes),
        int(min_tacks),
        int(min_gybes),
        float(tack_penalty_seconds),
        float(gybe_penalty_seconds),
        float(maneuver_separation_seconds),
        routing_model,
        float(timestep),
        float(spatial_resolution),
        source_id,
        current_id,
    )


initialize_state()
st.set_page_config(page_title="J/70 Router · Local Development", layout="wide")
st.markdown(
    """
    <style>
    .stButton button { min-height: 2.75rem; }
    @media (max-width: 768px) {
      .block-container { padding: 0.65rem 0.5rem 2.5rem; }
      h1 { font-size: 1.45rem !important; margin-bottom: 0.1rem !important; }
      h2, h3 { font-size: 1.12rem !important; }
      div[data-testid="stMetric"] { padding: 0.35rem 0; }
      button[data-baseweb="tab"] { min-height: 44px; padding-left: 0.7rem; padding-right: 0.7rem; }
      iframe[title="streamlit_folium.st_folium"] {
        height: 68svh !important;
        min-height: 460px !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("J/70 ROUTER — LOCAL DEVELOPMENT")
st.caption("Interactive Cascais course, active forecast, J/70 polar, and maneuver-constrained routing")

map_tab, polar_tab, weather_tab = st.tabs(["MAP", "POLAR", "WEATHER"])

with map_tab:
    selected_time = st.session_state.selected_forecast_time
    frame, leeward_xy, windward_xy, leeward_latlon, windward_latlon = geometry()
    length_m = distance(leeward_xy, windward_xy)
    bearing = course_bearing(leeward_xy, windward_xy)

    course_panel, weather_panel, sailing_panel = st.tabs(["COURSE", "WEATHER", "SAILING"])
    with course_panel:
        st.subheader("Course")
        st.radio("Select:", MARK_OPTIONS, key="selected_mark")
        st.caption("Choose a mark, then click its approximate location on the map. Correct it exactly below the map.")
        st.metric("Course length", f"{length_m:.0f} m")
        st.metric("Course bearing", f"{bearing:.1f}°")
        st.caption(
            "SI Appendix A, Course 1 approximation: two windward/leeward laps (four routed legs). "
            "The offset mark and two-mark gate are represented by the W/L axis endpoints for now."
        )
        if st.button("SET MARKS TO AREA 5", use_container_width=True):
            st.session_state.leeward_lat, st.session_state.leeward_lon = DEFAULT_LEEWARD
            st.session_state.windward_lat, st.session_state.windward_lon = DEFAULT_WINDWARD
            st.session_state.map_center = AREA_5_CENTER
            st.session_state.map_zoom = 13
            invalidate_route()
            st.rerun()

    with weather_panel:
        st.subheader("Weather")
        requested_source = st.selectbox(
            "Weather source to load (active source shown below)",
            ("IPMA Portugal · 2.5 km", "ICON-EU", "Synthetic"),
            index={"ipma_portugal": 0, "icon_eu": 1, "synthetic": 2}.get(
                st.session_state.requested_weather_source, 1
            ),
            key="weather_source_selector",
        )
        st.session_state.requested_weather_source = {
            "IPMA Portugal · 2.5 km": "ipma_portugal",
            "ICON-EU": "icon_eu",
            "Synthetic": "synthetic",
        }[requested_source]
        load_button, cache_button = st.columns(2)
        load_weather = load_button.button("LOAD / REFRESH WEATHER", type="primary", use_container_width=True)
        use_cache = cache_button.button("USE CACHED WEATHER", use_container_width=True)
        south, north, west, east = download_bounds(frame, leeward_xy, windward_xy)
        course_bbox = (
            min(leeward_latlon[0], windward_latlon[0]),
            max(leeward_latlon[0], windward_latlon[0]),
            min(leeward_latlon[1], windward_latlon[1]),
            max(leeward_latlon[1], windward_latlon[1]),
        )

        def activate_weather(source, source_key: str, label: str) -> None:
            latest_valid = max(source.grid.times)
            if source_key == "ipma_portugal" and latest_valid <= datetime.now(UTC):
                active = st.session_state.forecast_dataset
                active_name = active.model_name if active is not None else "Synthetic/test wind"
                st.session_state.weather_message = (
                    f"IPMA Portugal DOWNLOAD_FAILED: its newest valid time is "
                    f"{latest_valid:%Y-%m-%d %H:%M UTC}, so it contains no forward forecast. "
                    f"Active weather remains {active_name}."
                )
                return
            validation_time = min(
                source.grid.times,
                key=lambda value: abs((value - selected_time).total_seconds()),
            )
            validation = validate_weather_dataset(source, course_bbox, validation_time)
            if validation.status is WeatherSourceStatus.AVAILABLE:
                st.session_state.forecast_dataset = source
                st.session_state.loaded_weather_run = source.forecast_run
                st.session_state.weather_source = source_key
                select_nearest_forecast_time(source.grid.times)
                st.session_state.weather_message = label
            else:
                active = st.session_state.forecast_dataset
                active_name = active.model_name if active is not None else "Synthetic/test wind"
                st.session_state.weather_message = (
                    f"{source.model_name} {validation.status}: {validation.message}. "
                    f"Active weather remains {active_name}."
                )

        if load_weather:
            if st.session_state.requested_weather_source == "synthetic":
                st.session_state.weather_source = "synthetic"
                st.session_state.forecast_dataset = None
                st.session_state.loaded_weather_run = None
                st.session_state.weather_message = "Using synthetic/test wind by request."
            elif st.session_state.requested_weather_source == "icon_eu" and not os.environ.get("GRIBSTREAM_API_KEY"):
                st.session_state.weather_message = "ICON-EU cannot refresh: GRIBSTREAM_API_KEY is not configured."
            elif st.session_state.requested_weather_source == "ipma_portugal":
                try:
                    with st.spinner("Downloading the latest IPMA Portugal AROME-PT2 2.5 km wind…"):
                        run = get_latest_ipma_portugal_run()
                        source = download_ipma_portugal_box(
                            south, north, west, east, run, run + timedelta(hours=48), forecast_run=run
                        )
                    activate_weather(
                        source,
                        "ipma_portugal",
                        "Latest IPMA Portugal 2.5 km forecast loaded and validated.",
                    )
                except Exception as exc:
                    active = st.session_state.forecast_dataset
                    active_name = active.model_name if active is not None else "Synthetic/test wind"
                    st.session_state.weather_message = (
                        f"IPMA Portugal DOWNLOAD_FAILED: {exc}. "
                        f"Active weather remains {active_name}."
                    )
            else:
                try:
                    with st.spinner("Downloading the latest ICON-EU ~7 km wind subset…"):
                        run = get_latest_icon_eu_run()
                        source = download_icon_eu_box(
                            south, north, west, east, run, run + timedelta(hours=51), forecast_run=run
                        )
                    activate_weather(
                        source, "icon_eu", "Latest ICON-EU forecast loaded and validated."
                    )
                except Exception as exc:
                    st.session_state.weather_message = f"ICON-EU DOWNLOAD_FAILED: {exc}"

        if use_cache:
            try:
                if st.session_state.requested_weather_source == "ipma_portugal":
                    activate_weather(
                        load_cached_ipma_portugal(),
                        "ipma_portugal",
                        "Cached IPMA Portugal forecast loaded and validated.",
                    )
                elif st.session_state.requested_weather_source == "icon_eu":
                    activate_weather(
                        load_cached_icon_eu(), "icon_eu", "Cached ICON-EU forecast loaded and validated."
                    )
                else:
                    raise ValueError("Synthetic wind does not need a cache")
            except Exception as exc:
                st.session_state.weather_message = f"Cached weather unavailable: {exc}"

        # This selector only operates on the active, already-loaded dataset.
        # It is rendered after source actions so its options are immediately current.
        forecast_times = available_forecast_times()
        if st.session_state.selected_forecast_time not in forecast_times:
            select_forecast_time(forecast_times[0])
        now_column, forecast_column = st.columns([0.32, 0.68])
        if now_column.button("NOW model forecast", use_container_width=True):
            select_forecast_time(
                min(forecast_times, key=lambda value: abs((value - datetime.now(UTC)).total_seconds()))
            )
            invalidate_route()
        selected_time = forecast_column.selectbox(
            "Forecast time (Cascais local)",
            forecast_times,
            index=forecast_times.index(st.session_state.selected_forecast_time),
            format_func=format_cascais_time,
            key="forecast_time_widget",
            on_change=invalidate_route,
        )
        st.session_state.selected_forecast_time = selected_time
        st.caption(
            "Times are Cascais local time (WET/WEST). NOW uses the model forecast valid "
            "closest to the current time; it is not an observation."
        )

        st.caption(
            "Cascais models: IPMA Portugal AROME-PT2 2.5 km, ICON-EU fallback, "
            "or Synthetic by request. AROME-France remains excluded because its "
            "populated footprint does not cover Cascais."
        )
        if st.session_state.weather_message:
            st.info(st.session_state.weather_message)

        source = st.session_state.forecast_dataset
        if st.session_state.weather_source != "synthetic" and source is not None:
            st.markdown(f"**Active weather: {source.model_name}**")
            st.write(f"Model run: {source.forecast_run:%Y-%m-%d %H:00 UTC}")
            st.write(
                f"Forecast range: {format_cascais_time(source.grid.times[0])} – "
                f"{format_cascais_time(source.grid.times[-1])}"
            )
            st.write(f"Grid resolution: {source.grid_resolution_deg:g}°")
        else:
            st.markdown("**Active weather: Synthetic/test wind**")
            st.write("Model run: N/A")
            st.write("Grid resolution: synthetic")

        st.markdown("**Ocean current · Copernicus Marine IBI**")
        current_a, current_b = st.columns(2)
        load_current = current_a.button("LOAD / REFRESH CURRENT", use_container_width=True)
        use_cached_current = current_b.button("USE CACHED CURRENT", use_container_width=True)
        if load_current:
            if not copernicus_credentials_configured():
                st.session_state.current_message = (
                    "Copernicus credentials are not configured. Set "
                    "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD, "
                    "or run 'copernicusmarine login'."
                )
            else:
                try:
                    with st.spinner("Downloading the Copernicus IBI hourly surface-current subset…"):
                        st.session_state.current_dataset = download_current_box(
                            south,
                            north,
                            west,
                            east,
                            selected_time - timedelta(hours=1),
                            selected_time + timedelta(hours=12),
                        )
                    st.session_state.current_message = "Copernicus Marine current loaded and cached."
                    invalidate_route()
                except Exception as exc:
                    st.session_state.current_message = f"Copernicus current load failed: {exc}"
        if use_cached_current:
            try:
                st.session_state.current_dataset = load_cached_current()
                st.session_state.current_message = "Cached Copernicus Marine current loaded."
                invalidate_route()
            except Exception as exc:
                st.session_state.current_message = f"Cached Copernicus current unavailable: {exc}"
        st.caption(st.session_state.current_message)
        if st.session_state.current_dataset is not None:
            current_data = st.session_state.current_dataset
            st.write(f"Dataset: `{current_data.dataset_id}`")
            st.write(
                f"Current range: {format_cascais_time(current_data.grid.times[0])} – "
                f"{format_cascais_time(current_data.grid.times[-1])}"
            )
            st.write(f"Current grid: {current_data.grid_resolution_deg:g}°")

    if st.session_state.weather_source != "synthetic" and st.session_state.forecast_dataset is not None:
        arome_source = st.session_state.forecast_dataset
        wind = arome_source.for_local_frame(frame)
    else:
        arome_source = None
        wind = SyntheticWindField(
            base_speed_mps=5.2,
            base_direction_deg=bearing,
            speed_gradient_x_per_m=-0.00012,
            direction_gradient_x_deg_per_m=0.002,
            reference_time=selected_time,
        )
    raw_routing_current = (
        st.session_state.current_dataset.for_local_frame(frame)
        if st.session_state.current_dataset is not None else None
    )
    routing_current = (
        TimeClampedCurrentField(
            raw_routing_current,
            st.session_state.current_dataset.grid.times[0],
            st.session_state.current_dataset.grid.times[-1],
        )
        if raw_routing_current is not None else None
    )

    with sailing_panel:
        st.subheader("Sailing")
        sail_a, sail_b = st.columns(2)
        crew_factor = sail_a.number_input(
            "Crew speed factor", 0.50, 1.10, step=0.01, key="crew_speed_factor"
        )
        heading_sigma = sail_b.selectbox(
            "Heading sigma", (0, 2, 3, 5), format_func=lambda value: f"{value}°", key="heading_sigma_deg"
        )
        sail_c, sail_d = st.columns(2)
        max_tacks = sail_c.number_input("Maximum tacks", 0, 10, step=1, key="maximum_tacks")
        max_gybes = sail_d.number_input("Maximum gybes", 0, 10, step=1, key="maximum_gybes")
        sail_e, sail_f = st.columns(2)
        min_tacks = sail_e.number_input("Minimum tacks", 0, 10, step=1, key="minimum_tacks")
        min_gybes = sail_f.number_input("Minimum gybes", 0, 10, step=1, key="minimum_gybes")
        sail_g, sail_h = st.columns(2)
        tack_penalty_seconds = sail_g.number_input(
            "Tack penalty (s)", 0.0, 120.0, step=1.0, key="tack_penalty_seconds"
        )
        gybe_penalty_seconds = sail_h.number_input(
            "Gybe penalty (s)", 0.0, 120.0, step=1.0, key="gybe_penalty_seconds"
        )
        routing_model_label = st.selectbox(
            "Mode", ("Robust", "Theoretical"), key="routing_model_label"
        )
        routing_model = routing_model_label.lower()
        with st.expander("Advanced"):
            timestep = st.number_input(
                "Routing timestep (s)", 2.0, 30.0, step=1.0, key="routing_timestep_seconds"
            )
            spatial_resolution = st.number_input(
                "Spatial-state resolution (m)", 10.0, 100.0, step=5.0, key="spatial_resolution_m"
            )
            maneuver_separation_seconds = st.number_input(
                "Minimum maneuver separation (s)", 0.0, 120.0, step=1.0,
                key="maneuver_separation_seconds",
            )

        signature = make_route_signature(
            selected_time, crew_factor, int(heading_sigma), int(max_tacks), int(max_gybes),
            int(min_tacks), int(min_gybes), float(tack_penalty_seconds), float(gybe_penalty_seconds),
            float(maneuver_separation_seconds),
            routing_model, timestep, spatial_resolution,
        )
        calculate_route = st.button("CALCULATE ROUTE", type="primary", use_container_width=True)
        if calculate_route:
            started = time.perf_counter()
            try:
                base_routing_config = RoutingConfig(
                    dt_seconds=float(timestep),
                    spatial_bin_m=float(spatial_resolution),
                    mark_radius_m=15.0,
                    beam_width=400,
                    max_leg_time_seconds=7_200.0,
                    minimum_maneuver_separation_seconds=float(maneuver_separation_seconds),
                    max_computation_seconds=None,
                )
                routing_config = adaptive_routing_config(base_routing_config, length_m)
                for mark_name, mark_position in (
                    ("Leeward mark", leeward_xy),
                    ("Windward mark", windward_xy),
                ):
                    try:
                        wind.wind_at(mark_position, selected_time)
                    except WindOutOfBoundsError as exc:
                        raise RouteNotFoundError(
                            f"{mark_name} is outside the loaded weather subset. "
                            "Open WEATHER and press LOAD / REFRESH WEATHER after moving "
                            f"the marks. Details: {exc}"
                        ) from exc
                polar = get_polar(POLAR_DATA_VERSION)
                target_angle_model = TargetAngleModel(
                    polar, routing_model, float(heading_sigma)
                )
                estimated_leg_seconds = min(
                    routing_config.max_leg_time_seconds,
                    max(900.0, length_m / 1.5),
                )
                active_forecast = st.session_state.forecast_dataset
                estimated_finish = selected_time + timedelta(seconds=4.0 * estimated_leg_seconds)
                routing_wind = wind
                if (
                    st.session_state.weather_source != "synthetic"
                    and active_forecast is not None
                ):
                    routing_wind = TimeClampedWindField(
                        wind,
                        active_forecast.grid.times[0],
                        active_forecast.grid.times[-1],
                    )
                    if estimated_finish > active_forecast.grid.times[-1]:
                        st.warning(
                            "The race may extend beyond the loaded forecast. Routing will use "
                            f"the final forecast frame ({format_cascais_time(active_forecast.grid.times[-1])}) "
                            "for any remaining race time."
                        )
                if routing_current is not None:
                    try:
                        for current_position in (leeward_xy, windward_xy):
                            routing_current.current_at(current_position, selected_time)
                            routing_current.current_at(current_position, estimated_finish)
                    except WindOutOfBoundsError as exc:
                        raise RouteNotFoundError(
                            "The loaded Copernicus current does not cover the full course/time. "
                            "Open WEATHER and press LOAD / REFRESH CURRENT after setting the "
                            f"marks and forecast time. Details: {exc}"
                        ) from exc
                    current_grid = st.session_state.current_dataset.grid
                    if estimated_finish > current_grid.times[-1]:
                        st.warning(
                            "The race may extend beyond the loaded current forecast. Routing will "
                            f"hold the final current frame ({format_cascais_time(current_grid.times[-1])}) constant."
                        )
                upwind_reachable, upwind_margin = sampled_course_reachability(
                    leeward_xy,
                    windward_xy,
                    selected_time,
                    routing_wind,
                    target_angle_model,
                    "upwind",
                    estimated_leg_seconds,
                )
                if not upwind_reachable:
                    raise RouteNotFoundError(
                        "The windward mark is outside the legal upwind-heading sector "
                        f"in the sampled forecast (by about {-upwind_margin:.1f}\N{DEGREE SIGN}). "
                        "Move W farther toward the displayed TWD, swap the marks if they are "
                        "reversed, or select a forecast time with a suitable wind direction."
                    )
                downwind_start_time = selected_time + timedelta(seconds=estimated_leg_seconds)
                downwind_reachable, downwind_margin = sampled_course_reachability(
                    windward_xy,
                    leeward_xy,
                    downwind_start_time,
                    routing_wind,
                    target_angle_model,
                    "downwind",
                    estimated_leg_seconds,
                )
                if not downwind_reachable:
                    raise RouteNotFoundError(
                        "The return mark is outside the legal downwind-heading sector "
                        f"in the sampled forecast (by about {-downwind_margin:.1f}\N{DEGREE SIGN}). "
                        "Align the course more closely with the displayed TWD or select a "
                        "forecast time with a suitable wind direction."
                    )
                with st.spinner("Searching routes…"):
                    race = route_laps(
                        leeward_xy,
                        windward_xy,
                        selected_time,
                        routing_wind,
                        J70SpeedModel(polar, crew_speed_factor=crew_factor),
                        laps=2,
                        max_tacks=int(max_tacks),
                        max_gybes=int(max_gybes),
                        minimum_tacks=int(min_tacks),
                        minimum_gybes=int(min_gybes),
                        tack_penalty_seconds=float(tack_penalty_seconds),
                        gybe_penalty_seconds=float(gybe_penalty_seconds),
                        target_model=target_angle_model,
                        config=routing_config,
                        current=routing_current,
                    )
                st.session_state.last_route = race
                st.session_state.last_route_signature = signature
                st.session_state.last_route_execution_seconds = time.perf_counter() - started
                st.session_state.last_route_error = None
            except (RouteNotFoundError, WindOutOfBoundsError, ValueError) as exc:
                st.session_state.last_route_error = str(exc)
                st.session_state.last_route_execution_seconds = time.perf_counter() - started
        if st.session_state.last_route_error:
            st.error(st.session_state.last_route_error)

    current_route = (
        st.session_state.last_route
        if st.session_state.last_route is not None and st.session_state.last_route_signature == signature
        else None
    )
    if st.session_state.last_route is not None and current_route is None:
        st.info("Course, forecast time, or sailing settings changed. Recalculate to update the route overlay.")

    st.markdown("#### Interactive course and wind map")
    field_column, explanation_column = st.columns([0.24, 0.76])
    map_field = field_column.radio(
        "Map field", ("Wind", "Current"), horizontal=True, key="map_field"
    )
    current_source = st.session_state.current_dataset
    current_time = None
    if current_source is not None:
        current_time = min(
            current_source.grid.times,
            key=lambda value: abs((value - selected_time).total_seconds()),
        )
    if map_field == "Wind":
        explanation_column.caption(
            "Colors show wind speed. Arrows point where the air is moving; numeric TWD is the direction from which it comes."
        )
    elif current_source is None:
        explanation_column.warning(
            "Load Copernicus Marine current in the WEATHER settings. Course and route layers remain visible."
        )
    else:
        explanation_column.caption(
            "Colors show surface-current speed; arrows point where the water moves. "
            f"Valid {format_cascais_time(current_time)}."
        )
        current_age_hours = abs((selected_time - current_time).total_seconds()) / 3600.0
        if current_age_hours > 6.0:
            explanation_column.warning(
                f"Nearest available current is {current_age_hours:.0f} hours from the selected wind time."
            )
    course_map = build_course_map(
        leeward_latlon,
        windward_latlon,
        frame,
        leeward_xy,
        windward_xy,
        wind,
        selected_time,
        arome_source=arome_source,
        race=current_route,
        # Keep the serialized Folium script independent of user pan/zoom. The
        # component receives the live viewport separately below.
        centre=None,
        zoom=12,
        show_wind=map_field == "Wind",
        current_source=current_source,
        current_time=current_time,
    )

    def capture_map_interaction() -> None:
        # streamlit-folium copies the new component value to ``course_map``
        # before invoking this callback, so the next script render starts from
        # the new viewport instead of briefly restoring the previous one.
        process_map_interaction(st.session_state, st.session_state.get("course_map"))

    st_folium(
        course_map,
        key="course_map",
        height=680,
        use_container_width=True,
        center=tuple(st.session_state.map_center),
        zoom=int(st.session_state.map_zoom),
        returned_objects=["last_clicked", "last_object_clicked_tooltip", "center", "zoom"],
        on_change=capture_map_interaction,
    )

    coord_left, coord_right = st.columns(2)
    with coord_left:
        st.markdown("**LEEWARD MARK**")
        leeward_coord_a, leeward_coord_b = st.columns(2)
        leeward_coord_a.number_input(
            "Latitude", format="%.6f", step=0.000001, key="leeward_lat", on_change=recenter_on_marks
        )
        leeward_coord_b.number_input(
            "Longitude", format="%.6f", step=0.000001, key="leeward_lon", on_change=recenter_on_marks
        )
    with coord_right:
        st.markdown("**WINDWARD MARK**")
        windward_coord_a, windward_coord_b = st.columns(2)
        windward_coord_a.number_input(
            "Latitude", format="%.6f", step=0.000001, key="windward_lat", on_change=recenter_on_marks
        )
        windward_coord_b.number_input(
            "Longitude", format="%.6f", step=0.000001, key="windward_lon", on_change=recenter_on_marks
        )

    try:
        leeward_wind = wind.wind_at(leeward_xy, selected_time)
        windward_wind = wind.wind_at(windward_xy, selected_time)
        delta_speed = (windward_wind.speed_mps - leeward_wind.speed_mps) / KNOT_TO_MPS
        delta_direction = ((windward_wind.direction_deg - leeward_wind.direction_deg + 180) % 360) - 180
        info_course, info_weather, info_route = st.columns(3)
        with info_course:
            st.markdown("#### Course")
            st.metric("Length", f"{length_m:.0f} m")
            st.metric("Bearing", f"{bearing:.1f}°")
        with info_weather:
            st.markdown("#### Weather")
            if arome_source is not None:
                st.write(f"Model: {arome_source.model_name}")
                st.write(f"Run: {arome_source.forecast_run:%Y-%m-%d %H:00 UTC}")
            else:
                st.write("Model: Synthetic/test wind")
            st.write(f"Valid: {format_cascais_time(selected_time)}")
            weather_values = st.columns(2)
            weather_values[0].metric("Leeward TWS", f"{leeward_wind.speed_mps / KNOT_TO_MPS:.1f} kt")
            weather_values[0].metric("Leeward TWD", f"{leeward_wind.direction_deg:.0f}°")
            weather_values[1].metric("Windward TWS", f"{windward_wind.speed_mps / KNOT_TO_MPS:.1f} kt")
            weather_values[1].metric("Windward TWD", f"{windward_wind.direction_deg:.0f}°")
            st.write(f"ΔTWS (windward − leeward): {delta_speed:+.2f} kt")
            st.write(f"ΔTWD (windward − leeward): {delta_direction:+.1f}°")
            if map_field == "Current" and current_source is not None and current_time is not None:
                try:
                    leeward_current = current_source.current_at(*leeward_latlon, current_time)
                    windward_current = current_source.current_at(*windward_latlon, current_time)
                    current_values = st.columns(2)
                    current_values[0].metric("Leeward current", f"{leeward_current['speed_knots']:.2f} kt")
                    current_values[0].caption(f"toward {leeward_current['direction_to_deg']:.0f}°")
                    current_values[1].metric("Windward current", f"{windward_current['speed_knots']:.2f} kt")
                    current_values[1].caption(f"toward {windward_current['direction_to_deg']:.0f}°")
                except (WindOutOfBoundsError, ValueError) as exc:
                    st.warning(f"A mark is outside the loaded current subset: {exc}")
        with info_route:
            st.markdown("#### Route")
            if current_route is None:
                st.write("No current route. Press **CALCULATE ROUTE**.")
            else:
                st.metric("Race time", format_duration(current_route.total_elapsed_seconds))
                route_values = st.columns(3)
                route_values[0].metric("Tacks", current_route.tacks)
                route_values[1].metric("Gybes", current_route.gybes)
                route_values[2].metric("Distance", f"{current_route.total_distance_m / 1852:.2f} NM")
                average_tws = (
                    leeward_wind.speed_mps + windward_wind.speed_mps
                ) / (2.0 * KNOT_TO_MPS)
                display_targets = TargetAngleModel(
                    get_polar(POLAR_DATA_VERSION), routing_model, float(heading_sigma)
                )
                upwind_target = display_targets(average_tws * KNOT_TO_MPS, "upwind")
                downwind_target = display_targets(average_tws * KNOT_TO_MPS, "downwind")
                target_values = st.columns(3)
                target_values[0].metric("Average TWS", f"{average_tws:.1f} kt")
                target_values[1].metric("Upwind target TWA", f"{upwind_target:.1f}°")
                target_values[2].metric(
                    "Downwind target TWA",
                    f"{downwind_target:.1f}°",
                    help=f"Approximate heading change through a gybe: {360.0 - 2.0 * downwind_target:.1f}°",
                )
                st.caption(
                    "Copernicus current included in routing."
                    if routing_current is not None else "Wind-only routing; no current dataset loaded."
                )
                tactical_rows = []
                for lap_index, lap in enumerate(current_route.lap_results, start=1):
                    for leg_name, options in (
                        ("Upwind", lap.upwind_baselines),
                        ("Downwind", lap.downwind_baselines),
                    ):
                        if options:
                            fastest = min(option.route.elapsed_seconds for option in options)
                            for option in options:
                                tactical_rows.append(
                                    {
                                        "Leg": f"Lap {lap_index} {leg_name}",
                                        "First move": option.course_side.title(),
                                        "Initial tack/gybe": option.initial_side.title(),
                                        "One-maneuver time": format_duration(option.route.elapsed_seconds),
                                        "Difference": f"{option.route.elapsed_seconds - fastest:+.0f} s",
                                    }
                                )
                if tactical_rows:
                    st.markdown("**Left/right baseline · one maneuver**")
                    st.dataframe(tactical_rows, hide_index=True, use_container_width=True)
    except WindOutOfBoundsError as exc:
        st.warning(f"A mark is outside the loaded forecast subset: {exc}")

    if current_route is not None:
        with st.expander("DEBUG"):
            debug_a, debug_b, debug_c = st.columns(3)
            debug_a.metric("Route states evaluated", f"{current_route.states_evaluated:,}")
            debug_b.metric("Optimizer execution", f"{st.session_state.last_route_execution_seconds:.3f} s")
            debug_c.metric(
                "Selected model run",
                st.session_state.loaded_weather_run.strftime("%Y-%m-%d %H:%M UTC")
                if st.session_state.loaded_weather_run else "Synthetic",
            )
            events = []
            for leg in current_route.legs:
                event_type = "tack" if leg.mode == "upwind" else "gybe"
                for maneuver in leg.maneuvers:
                    lat, lon = frame.to_latlon(maneuver.x, maneuver.y)
                    events.append(
                        {
                            "type": event_type,
                            "time_utc": maneuver.time.isoformat(),
                            "latitude": round(lat, 7),
                            "longitude": round(lon, 7),
                            "change": f"{maneuver.from_side} → {maneuver.to_side}",
                        }
                    )
            st.dataframe(events, use_container_width=True)

with polar_tab:
    st.subheader("J/70 polar development")
    controls = st.columns(3)
    polar_tws = controls[0].slider("TWS (kt)", 4.0, 20.0, 10.0, 0.1, key="polar_tws")
    polar_sigma = controls[1].selectbox(
        "Heading sigma", (0, 2, 3, 5), index=2, format_func=lambda value: f"{value}°", key="polar_sigma"
    )
    polar_crew = controls[2].slider("Crew speed factor", 0.50, 1.05, 0.93, 0.01, key="polar_crew")
    polar = get_polar(POLAR_DATA_VERSION)
    st.caption(
        f"{polar.metadata['name']} · {polar.metadata['sail_configuration']} · "
        f"VPP {polar.metadata['vpp_version']}"
    )
    angles = np.linspace(30, 180, 601)
    polar_figure, polar_axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for wind_speed in (6, 8, 10, 12, 14, 16, 20):
        polar_axis.plot(
            angles,
            [polar.boat_speed(wind_speed, angle) * polar_crew for angle in angles],
            label=f"{wind_speed} kt",
        )
    polar_axis.set(
        xlabel="True wind angle (deg)",
        ylabel="Boat speed after crew factor (kt)",
        title="J/70 ORC best-performance polar",
    )
    polar_axis.grid(alpha=0.22)
    polar_axis.legend(title="TWS", ncols=2)
    st.pyplot(polar_figure, use_container_width=True)
    plt.close(polar_figure)

    theory_up = target_twa(polar, polar_tws, "upwind", "theoretical", 0)
    robust_up = target_twa(polar, polar_tws, "upwind", "robust", float(polar_sigma))
    theory_down = target_twa(polar, polar_tws, "downwind", "theoretical", 0)
    robust_down = target_twa(polar, polar_tws, "downwind", "robust", float(polar_sigma))
    target_metrics = st.columns(4)
    target_metrics[0].metric("Theoretical upwind target", f"{theory_up:.1f}°")
    target_metrics[1].metric("Robust upwind target", f"{robust_up:.1f}°")
    target_metrics[2].metric("Theoretical downwind target", f"{theory_down:.1f}°")
    target_metrics[3].metric("Robust downwind target", f"{robust_down:.1f}°")
    st.metric("Robust heading change through gybe", f"{360.0 - 2.0 * robust_down:.1f}°")

    vmg_figure, vmg_axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, mode, angle_range, theoretical_angle, robust_angle in (
        (vmg_axes[0], "upwind", np.linspace(25, 90, 326), theory_up, robust_up),
        (vmg_axes[1], "downwind", np.linspace(90, 180, 451), theory_down, robust_down),
    ):
        perfect = [expected_vmg_knots(polar, polar_tws, angle, mode, 0) * polar_crew for angle in angle_range]
        expected = [
            expected_vmg_knots(polar, polar_tws, angle, mode, float(polar_sigma)) * polar_crew
            for angle in angle_range
        ]
        axis.plot(angle_range, perfect, label="theoretical", color="#2474b5")
        axis.plot(angle_range, expected, label=f"robust σ={polar_sigma}°", color="#e67e22")
        axis.axvline(theoretical_angle, color="#2474b5", linestyle=":")
        axis.axvline(robust_angle, color="#e67e22", linestyle=":")
        axis.set(
            xlabel="Commanded TWA (deg)", ylabel="VMG (kt)",
            title=f"{mode.capitalize()} VMG at {polar_tws:.1f} kt TWS",
        )
        axis.grid(alpha=0.22)
        axis.legend()
    st.pyplot(vmg_figure, use_container_width=True)
    plt.close(vmg_figure)
    st.caption(
        "This ORC polar explicitly specifies an asymmetric spinnaker on centerline. "
        "Below 6 kt the current table clamps to its 6 kt row. Crew speed factor scales "
        "the curves but does not change either target angle."
    )

with weather_tab:
    st.subheader("Interpolated weather inspection")
    frame, leeward_xy, windward_xy, _, _ = geometry()
    selected_time = st.session_state.selected_forecast_time
    source = st.session_state.forecast_dataset
    if st.session_state.weather_source != "synthetic" and source is not None:
        inspection_wind = source.for_local_frame(frame)
        st.write(
            f"{source.model_name} run: {source.forecast_run:%Y-%m-%d %H:00 UTC} · "
            f"valid: {format_cascais_time(selected_time)}"
        )
    else:
        inspection_wind = SyntheticWindField(
            base_speed_mps=5.2,
            base_direction_deg=course_bearing(leeward_xy, windward_xy),
            speed_gradient_x_per_m=-0.00012,
            direction_gradient_x_deg_per_m=0.002,
            reference_time=selected_time,
        )
        st.write(f"Synthetic/test wind · valid: {format_cascais_time(selected_time)}")
    try:
        weather_figure = plot_wind_field_map(
            leeward_xy, windward_xy, inspection_wind, selected_time, padding_m=900.0
        )
        st.pyplot(weather_figure, use_container_width=True)
        plt.close(weather_figure)
    except WindOutOfBoundsError as exc:
        st.warning(f"Wind plot is outside the loaded forecast subset: {exc}")

    point_a, point_b = st.columns(2)
    inspect_lat = point_a.number_input(
        "Inspection latitude", value=(st.session_state.leeward_lat + st.session_state.windward_lat) / 2,
        format="%.6f", step=0.000001,
    )
    inspect_lon = point_b.number_input(
        "Inspection longitude", value=(st.session_state.leeward_lon + st.session_state.windward_lon) / 2,
        format="%.6f", step=0.000001,
    )
    try:
        sample = inspection_wind.wind_at(frame.to_xy(inspect_lat, inspect_lon), selected_time)
        metrics = st.columns(4)
        metrics[0].metric("U", f"{sample.u_mps:.3f} m/s")
        metrics[1].metric("V", f"{sample.v_mps:.3f} m/s")
        metrics[2].metric("TWS", f"{sample.speed_mps / KNOT_TO_MPS:.2f} kt")
        metrics[3].metric("TWD", f"{sample.direction_deg:.1f}°")
    except WindOutOfBoundsError as exc:
        st.warning(f"Inspection point is outside the loaded forecast subset: {exc}")
