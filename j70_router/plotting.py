"""Route and wind-field visualizations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .routing.optimizer import MultiLapResult, RaceResult, RouteResult
from .wind.base import WindField

SIDE_COLOURS = {"port": "#2474b5", "starboard": "#e67e22"}


def _bounds(
    race: RaceResult,
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    padding_m: float,
) -> tuple[float, float, float, float]:
    points = list(race.upwind.points) + list(race.downwind.points)
    xs = [p.x for p in points] + [downwind_mark[0], upwind_mark[0]]
    ys = [p.y for p in points] + [downwind_mark[1], upwind_mark[1]]
    return min(xs) - padding_m, max(xs) + padding_m, min(ys) - padding_m, max(ys) + padding_m


def _wind_grid(
    wind: WindField,
    sample_time: datetime,
    bounds: tuple[float, float, float, float],
    count: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, count)
    y = np.linspace(ymin, ymax, count)
    xx, yy = np.meshgrid(x, y)
    u = np.empty_like(xx)
    v = np.empty_like(yy)
    speed = np.empty_like(xx)
    for index in np.ndindex(xx.shape):
        sample = wind.wind_at((float(xx[index]), float(yy[index])), sample_time)
        u[index], v[index], speed[index] = sample.u_mps, sample.v_mps, sample.speed_mps
    return xx, yy, u, v, speed


def _plot_route_segments(ax: plt.Axes, route: RouteResult, linewidth: float = 2.4) -> None:
    for previous, current in zip(route.points, route.points[1:]):
        ax.plot(
            [previous.x, current.x],
            [previous.y, current.y],
            color=SIDE_COLOURS[current.side],
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=5,
        )


def _decorate_course(
    ax: plt.Axes,
    race: RaceResult,
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
) -> None:
    ax.plot(
        [downwind_mark[0], upwind_mark[0]],
        [downwind_mark[1], upwind_mark[1]],
        linestyle="--",
        color="0.35",
        linewidth=1.2,
        label="course axis",
        zorder=2,
    )
    _plot_route_segments(ax, race.upwind)
    _plot_route_segments(ax, race.downwind)
    ax.scatter(*downwind_mark, marker="o", s=85, color="#198754", edgecolor="white", zorder=7)
    ax.scatter(*upwind_mark, marker="*", s=180, color="#c62828", edgecolor="white", zorder=7)
    for maneuver in race.upwind.maneuvers:
        ax.scatter(maneuver.x, maneuver.y, marker="^", s=50, color="black", zorder=8)
    for maneuver in race.downwind.maneuvers:
        ax.scatter(maneuver.x, maneuver.y, marker="D", s=38, color="black", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East of downwind mark (m)")
    ax.set_ylabel("North of downwind mark (m)")
    ax.grid(alpha=0.18)


def plot_race_and_wind(
    race: RaceResult,
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    wind: WindField,
    sample_time: datetime,
    output_path: str | Path | None = None,
    padding_m: float = 250.0,
) -> plt.Figure:
    """Create a route panel and a separate wind-speed explanation panel."""

    bounds = _bounds(race, downwind_mark, upwind_mark, padding_m)
    xx, yy, u, v, speed = _wind_grid(wind, sample_time, bounds)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)

    axes[0].quiver(xx, yy, u, v, color="0.55", alpha=0.65, pivot="middle", zorder=1)
    _decorate_course(axes[0], race, downwind_mark, upwind_mark)
    axes[0].set_title("Optimized one-lap route")

    image = axes[1].pcolormesh(xx, yy, speed, shading="gouraud", cmap="viridis")
    axes[1].quiver(xx, yy, u, v, color="white", alpha=0.8, pivot="middle", zorder=3)
    _decorate_course(axes[1], race, downwind_mark, upwind_mark)
    axes[1].set_title(f"Why: wind at {sample_time:%Y-%m-%d %H:%M} UTC\n(arrows show air flow toward)")
    colourbar = figure.colorbar(image, ax=axes[1], shrink=0.78)
    colourbar.set_label("Wind speed (m/s)")

    handles = [
        Line2D([0], [0], color=SIDE_COLOURS["port"], lw=3, label="port segment"),
        Line2D([0], [0], color=SIDE_COLOURS["starboard"], lw=3, label="starboard segment"),
        Line2D([0], [0], marker="^", color="black", lw=0, label="tack"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="gybe"),
        Line2D([0], [0], marker="*", color="#c62828", lw=0, markersize=12, label="upwind mark"),
        Line2D([0], [0], marker="o", color="#198754", lw=0, label="downwind mark"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False)
    figure.suptitle(
        f"J/70 synthetic routing — {race.total_elapsed_seconds / 60:.1f} min, "
        f"{race.upwind.maneuvers_used} tacks, {race.downwind.maneuvers_used} gybes"
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure


def plot_route_wind_map(
    race: RaceResult | None,
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    wind: WindField,
    sample_time: datetime,
    padding_m: float = 400.0,
    title: str = "J/70 route and wind field",
) -> plt.Figure:
    """Single-panel development map rendered directly by Streamlit."""

    if race is None:
        xs = [downwind_mark[0], upwind_mark[0]]
        ys = [downwind_mark[1], upwind_mark[1]]
        bounds = min(xs) - padding_m, max(xs) + padding_m, min(ys) - padding_m, max(ys) + padding_m
    else:
        bounds = _bounds(race, downwind_mark, upwind_mark, padding_m)
    xx, yy, u, v, speed = _wind_grid(wind, sample_time, bounds, count=17)
    figure, ax = plt.subplots(figsize=(11.5, 8.0), constrained_layout=True)
    image = ax.pcolormesh(xx, yy, speed / 0.514444, shading="gouraud", cmap="viridis")
    ax.quiver(xx, yy, u, v, color="white", alpha=0.82, pivot="middle", zorder=3)
    ax.plot(
        [downwind_mark[0], upwind_mark[0]], [downwind_mark[1], upwind_mark[1]],
        linestyle="--", color="white", linewidth=1.4, alpha=0.9, zorder=4,
    )
    if race is not None:
        _plot_route_segments(ax, race.upwind, linewidth=3.0)
        _plot_route_segments(ax, race.downwind, linewidth=3.0)
        for maneuver in race.upwind.maneuvers:
            ax.scatter(maneuver.x, maneuver.y, marker="^", s=65, color="black", edgecolor="white", zorder=8)
        for maneuver in race.downwind.maneuvers:
            ax.scatter(maneuver.x, maneuver.y, marker="D", s=50, color="black", edgecolor="white", zorder=8)
    ax.scatter(*downwind_mark, marker="o", s=105, color="#21a366", edgecolor="white", linewidth=1.5, zorder=9)
    ax.scatter(*upwind_mark, marker="*", s=220, color="#d62728", edgecolor="white", linewidth=1.2, zorder=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="East (m)", ylabel="North (m)", title=title)
    ax.grid(alpha=0.16)
    colourbar = figure.colorbar(image, ax=ax, shrink=0.84)
    colourbar.set_label("True wind speed (kt)")
    handles = [
        Line2D([0], [0], color=SIDE_COLOURS["port"], lw=3, label="port"),
        Line2D([0], [0], color=SIDE_COLOURS["starboard"], lw=3, label="starboard"),
        Line2D([0], [0], ls="--", color="white", lw=2, label="course axis"),
        Line2D([0], [0], marker="^", color="black", markeredgecolor="white", lw=0, label="tack"),
        Line2D([0], [0], marker="D", color="black", markeredgecolor="white", lw=0, label="gybe"),
    ]
    ax.legend(handles=handles, loc="upper right", ncols=1, framealpha=0.85)
    return figure


def plot_wind_field_map(
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    wind: WindField,
    sample_time: datetime,
    padding_m: float = 800.0,
) -> plt.Figure:
    return plot_route_wind_map(
        None, downwind_mark, upwind_mark, wind, sample_time, padding_m,
        title=f"Wind field at {sample_time:%Y-%m-%d %H:%M} UTC",
    )


def plot_multilap_and_wind(
    race: MultiLapResult,
    downwind_mark: tuple[float, float],
    upwind_mark: tuple[float, float],
    wind: WindField,
    sample_time: datetime,
    output_path: str | Path | None = None,
    padding_m: float = 350.0,
) -> plt.Figure:
    all_points = [point for leg in race.legs for point in leg.points]
    xs = [p.x for p in all_points] + [downwind_mark[0], upwind_mark[0]]
    ys = [p.y for p in all_points] + [downwind_mark[1], upwind_mark[1]]
    bounds = min(xs) - padding_m, max(xs) + padding_m, min(ys) - padding_m, max(ys) + padding_m
    xx, yy, u, v, speed = _wind_grid(wind, sample_time, bounds)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)

    def decorate(ax: plt.Axes) -> None:
        ax.plot(
            [downwind_mark[0], upwind_mark[0]],
            [downwind_mark[1], upwind_mark[1]],
            linestyle="--",
            color="0.35",
            linewidth=1.2,
            zorder=2,
        )
        for leg_index, leg in enumerate(race.legs):
            lap = leg_index // 2
            for previous, current in zip(leg.points, leg.points[1:]):
                ax.plot(
                    [previous.x, current.x],
                    [previous.y, current.y],
                    color=SIDE_COLOURS[current.side],
                    linewidth=2.5,
                    linestyle="-" if lap == 0 else "--",
                    alpha=0.95 if lap == 0 else 0.8,
                    solid_capstyle="round",
                    zorder=5,
                )
            marker = "^" if leg.mode == "upwind" else "D"
            for maneuver in leg.maneuvers:
                ax.scatter(maneuver.x, maneuver.y, marker=marker, s=46, color="black", zorder=8)
        ax.scatter(*downwind_mark, marker="o", s=85, color="#198754", edgecolor="white", zorder=9)
        ax.scatter(*upwind_mark, marker="*", s=180, color="#c62828", edgecolor="white", zorder=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East of course centre (m)")
        ax.set_ylabel("North of course centre (m)")
        ax.grid(alpha=0.18)

    axes[0].quiver(xx, yy, u, v, color="0.55", alpha=0.65, pivot="middle", zorder=1)
    decorate(axes[0])
    axes[0].set_title("Optimized two-lap route")
    image = axes[1].pcolormesh(xx, yy, speed, shading="gouraud", cmap="viridis")
    axes[1].quiver(xx, yy, u, v, color="white", alpha=0.82, pivot="middle", zorder=3)
    decorate(axes[1])
    axes[1].set_title(f"Forecast at {sample_time:%Y-%m-%d %H:%M} UTC\n(arrows show air flow toward)")
    colourbar = figure.colorbar(image, ax=axes[1], shrink=0.78)
    colourbar.set_label("Wind speed (m/s)")
    handles = [
        Line2D([0], [0], color=SIDE_COLOURS["port"], lw=3, label="port"),
        Line2D([0], [0], color=SIDE_COLOURS["starboard"], lw=3, label="starboard"),
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="lap 1"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="lap 2"),
        Line2D([0], [0], marker="^", color="black", lw=0, label="tack"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="gybe"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False)
    figure.suptitle(
        f"Two-lap Cascais test — {race.total_elapsed_seconds / 60:.1f} min, "
        f"{race.tacks} tacks, {race.gybes} gybes"
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure
