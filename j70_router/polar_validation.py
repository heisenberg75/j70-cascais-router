"""Generate J/70 polar and robust-target sanity plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .sailing.j70_polar import J70Polar
from .sailing.targets import expected_vmg_knots, target_twa

OUTPUT_DIR = Path("artifacts/polar")
WINDS = (6, 8, 10, 12, 14, 16, 20)


def plot_polar(polar: J70Polar) -> None:
    angles = np.linspace(30, 180, 601)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    for wind in WINDS:
        ax.plot(angles, [polar.boat_speed(wind, a) for a in angles], label=f"{wind} kt")
    ax.set(xlabel="True wind angle (deg)", ylabel="Boat speed (kt)", title="J/70 ORC best-performance polar")
    ax.grid(alpha=0.25)
    ax.legend(title="TWS", ncols=2)
    fig.savefig(OUTPUT_DIR / "j70_polar.png", dpi=170)
    plt.close(fig)


def plot_vmg(polar: J70Polar) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, wind in zip(axes.flat, (8, 10, 12, 16)):
        angles = np.linspace(30, 75, 451)
        theoretical = [expected_vmg_knots(polar, wind, a, "upwind", 0) for a in angles]
        robust = [expected_vmg_knots(polar, wind, a, "upwind", 3) for a in angles]
        theory_angle = target_twa(polar, wind, "upwind", "theoretical")
        robust_angle = target_twa(polar, wind, "upwind", "robust", 3)
        ax.plot(angles, theoretical, label="perfect steering", color="#1f77b4")
        ax.plot(angles, robust, label="expected, sigma=3 deg", color="#d95f02")
        ax.axvline(theory_angle, color="#1f77b4", ls=":")
        ax.axvline(robust_angle, color="#d95f02", ls=":")
        ax.set_title(f"TWS {wind} kt: {theory_angle:.1f} deg / {robust_angle:.1f} deg")
        ax.set(xlabel="Commanded TWA (deg)", ylabel="Upwind VMG (kt)")
        ax.grid(alpha=0.22)
    axes[0, 0].legend()
    fig.suptitle("J/70 upwind VMG: theoretical and steering-robust target")
    fig.savefig(OUTPUT_DIR / "j70_vmg_robust.png", dpi=170)
    plt.close(fig)


def plot_targets(polar: J70Polar) -> None:
    winds = np.linspace(6, 20, 57)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for sigma in (0, 2, 3, 5):
        sailing_mode = "theoretical" if sigma == 0 else "robust"
        axes[0].plot(winds, [target_twa(polar, w, "upwind", sailing_mode, sigma) for w in winds], label=f"sigma={sigma} deg")
        axes[1].plot(winds, [target_twa(polar, w, "downwind", sailing_mode, sigma) for w in winds], label=f"sigma={sigma} deg")
    axes[0].set_title("Upwind target")
    axes[1].set_title("Downwind target")
    for ax in axes:
        ax.set(xlabel="TWS (kt)", ylabel="Commanded TWA (deg)")
        ax.grid(alpha=0.22)
        ax.legend()
    fig.suptitle("J/70 target TWA under heading uncertainty")
    fig.savefig(OUTPUT_DIR / "j70_target_twa.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    polar = J70Polar()
    plot_polar(polar)
    plot_vmg(polar)
    plot_targets(polar)
    print(f"wrote polar diagnostics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
