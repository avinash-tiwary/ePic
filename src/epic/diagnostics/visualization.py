"""
Publication-Quality Visualization & Animation Engine for PIC Simulations
========================================================================
Generates high-contrast scientific charts, multi-panel diagnostic dashboards,
and smooth animated GIFs / MP4 movies for 1D, 2D, and 3D kinetic plasma experiments.
Enforces the unified ePic deep-space theme across all renders.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from .style import apply_epic_style, EPIC_COLORS


def animate_1d_phase_space(
    snapshots: List[Dict[str, Any]],
    time_history: np.ndarray,
    energy_history: np.ndarray,
    output_path: str,
    boxsize: float,
    v_lim: float = 6.0,
    fps: int = 20,
    dpi: int = 120,
):
    """Create a publication-quality animation of 1D phase space evolution.

    Layout:
    - Left (large): Phase space scatter (x, vx) colored by beam/velocity
    - Top Right: Real-time velocity distribution histogram f(vx)
    - Bottom Right: Electric field energy vs time with moving progress marker
    """
    apply_epic_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = plt.figure(figsize=(14, 7), dpi=dpi, facecolor=EPIC_COLORS["bg_dark"])
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], figure=fig)

    ax_phase = fig.add_subplot(gs[:, 0], facecolor=EPIC_COLORS["bg_axes"])
    ax_dist = fig.add_subplot(gs[0, 1], facecolor=EPIC_COLORS["bg_axes"])
    ax_energy = fig.add_subplot(gs[1, 1], facecolor=EPIC_COLORS["bg_axes"])

    # Pre-plot energy history
    ax_energy.semilogy(time_history, np.maximum(energy_history, 1e-12), color=EPIC_COLORS["crimson"], lw=2.0, label=r"$\mathcal{E}_{field}(t)$")
    marker, = ax_energy.plot([], [], "o", color=EPIC_COLORS["gold"], markersize=8, label="Current Time")
    ax_energy.set_xlabel(r"Time ($\omega_{pe} t$)")
    ax_energy.set_ylabel(r"Field Energy $\mathcal{E}_{field}$")
    ax_energy.set_title("Electrostatic Field Energy Growth", fontsize=11, fontweight="bold")
    ax_energy.set_xlim(time_history[0], time_history[-1])
    ax_energy.legend(loc="lower right", fontsize=9)

    # Initial frame
    s0 = snapshots[0]
    p_x1, p_v1 = s0["beam1"]
    p_x2, p_v2 = s0["beam2"]

    sub = slice(None, None, max(1, len(p_x1) // 15000))
    sc1 = ax_phase.scatter(p_x1[sub], p_v1[sub], s=0.7, color=EPIC_COLORS["cyan"], alpha=0.5, label="Beam 1 (+v)")
    sc2 = ax_phase.scatter(p_x2[sub], p_v2[sub], s=0.7, color=EPIC_COLORS["crimson"], alpha=0.5, label="Beam 2 (-v)")
    ax_phase.set_xlim(0, boxsize)
    ax_phase.set_ylim(-v_lim, v_lim)
    ax_phase.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax_phase.set_ylabel(r"Velocity $v_x / v_{th}$")
    title_text = ax_phase.set_title(r"Phase Space: $\omega_{pe} t = 0.0$", fontweight="bold")
    ax_phase.legend(loc="upper right", markerscale=8, fontsize=9)

    # Histogram
    bins = np.linspace(-v_lim, v_lim, 60)
    all_v = np.concatenate([p_v1, p_v2])
    n_counts, _ = np.histogram(all_v, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dist_line, = ax_dist.plot(bin_centers, n_counts, color=EPIC_COLORS["emerald"], lw=2.2)
    ax_dist.set_xlim(-v_lim, v_lim)
    ax_dist.set_ylim(0, 0.4)
    ax_dist.set_xlabel(r"$v_x / v_{th}$")
    ax_dist.set_ylabel(r"$f(v_x)$")
    ax_dist.set_title("Velocity Distribution Function", fontsize=11, fontweight="bold")

    plt.tight_layout()

    def update(frame_idx):
        snap = snapshots[frame_idx]
        t = snap["time"]
        px1, pv1 = snap["beam1"]
        px2, pv2 = snap["beam2"]

        # Update phase space positions
        sc1.set_offsets(np.column_stack([px1[sub], pv1[sub]]))
        sc2.set_offsets(np.column_stack([px2[sub], pv2[sub]]))
        title_text.set_text(rf"Phase Space Vortex: $\omega_{{pe}} t = {t:.1f}$")

        # Update velocity distribution
        all_vel = np.concatenate([pv1, pv2])
        counts, _ = np.histogram(all_vel, bins=bins, density=True)
        dist_line.set_ydata(counts)

        # Update energy marker
        idx = np.argmin(np.abs(time_history - t))
        marker.set_data([time_history[idx]], [np.maximum(energy_history[idx], 1e-12)])

        return sc1, sc2, title_text, dist_line, marker

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps, blit=False)

    if output_path.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500, codec="libx264")
    else:
        writer = animation.PillowWriter(fps=fps)

    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"  [Movie Exported - EPIC STYLE] Successfully saved: {output_path}")


def animate_2d_density(
    density_frames: List[np.ndarray],
    time_points: List[float],
    output_path: str,
    Lx: float,
    Ly: float,
    fps: int = 15,
    dpi: int = 120,
):
    """Create animation of 2D charge density contours over time."""
    apply_epic_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi, facecolor=EPIC_COLORS["bg_dark"])
    ax.set_facecolor(EPIC_COLORS["bg_axes"])
    im = ax.imshow(
        density_frames[0],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="inferno",
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax, label=r"Charge Density $\rho(x, y)$")
    cbar.ax.yaxis.label.set_color(EPIC_COLORS["text"])
    ax.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax.set_ylabel(r"y ($c/\omega_{pe}$)")
    title = ax.set_title(r"2D Plasma Density: $\omega_{pe} t = 0.0$", fontweight="bold")

    plt.tight_layout()

    all_vals = np.array(density_frames)
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    im.set_clim(vmin, vmax)

    def update(i):
        im.set_data(density_frames[i])
        title.set_text(rf"2D Filamentation Density: $\omega_{{pe}} t = {time_points[i]:.1f}$")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(density_frames), interval=1000 // fps)
    if output_path.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500, codec="libx264")
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"  [Movie Exported - EPIC STYLE] Successfully saved: {output_path}")
