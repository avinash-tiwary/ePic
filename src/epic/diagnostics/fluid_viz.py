"""
Fluid-Moment and Continuous Phase-Space Fluid Visualization Engine
===================================================================
Transforms discrete Particle-In-Cell macroparticles into continuous,
silky-smooth hydrodynamic fluid fields by invoking Liouville's theorem
(incompressible phase-space flow df/dt = 0) and macroscopic kinetic moments.

Features:
- Continuous Vlasov Phase Fluid Reconstruction (f(x, vx) via 2D KDE / Gaussian filtering)
- Macroscopic Fluid Moments:
    * Density n(x, y)
    * Bulk flow velocity u(x, y)
    * Kinetic temperature / thermal pressure P(x, y)
    * Fluid vorticity omega_z = (nabla x u)_z
- Phase-space Hamiltonian streamlines & vortex vector fields
- Cinema-grade fluid animations (MP4 / GIF) with cosmic dark obsidian styling.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from .style import EPIC_COLORS, apply_epic_style, format_epic_figure, save_epic_plot


def reconstruct_phase_fluid(
    x: np.ndarray,
    vx: np.ndarray,
    boxsize: float,
    v_lim: float = 6.0,
    Nx_bins: int = 350,
    Nv_bins: int = 350,
    sigma: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct continuous phase-space distribution function f(x, vx) from particles.

    Applies fine 2D binning followed by Gaussian spatial smoothing to convert
    discrete macroparticles into a continuous, differentiable phase-space fluid density.

    Returns
    -------
    X_grid, V_grid : np.ndarray
        Meshgrid of phase-space coordinates.
    f_fluid : np.ndarray
        Continuous phase-space fluid density f(x, vx) of shape (Nv_bins, Nx_bins).
    """
    x = np.asarray(x).ravel()
    vx = np.asarray(vx).ravel()

    # Bounded particle range
    x_edges = np.linspace(0.0, boxsize, Nx_bins + 1)
    v_edges = np.linspace(-v_lim, v_lim, Nv_bins + 1)

    hist, _, _ = np.histogram2d(x, vx, bins=[x_edges, v_edges])

    # Periodic convolution in x, clamped in v
    # Wrap padding in x to respect periodic boundary conditions
    pad_x = 10
    hist_padded = np.pad(hist, ((pad_x, pad_x), (0, 0)), mode="wrap")
    hist_smooth = ndi.gaussian_filter(hist_padded, sigma=[sigma, sigma], mode="nearest")
    hist_clean = hist_smooth[pad_x:-pad_x, :]

    # Transpose so rows = velocity (y-axis) and cols = position (x-axis)
    f_fluid = hist_clean.T

    # Normalize to probability density integral f dx dv = 1
    dx = boxsize / Nx_bins
    dv = (2.0 * v_lim) / Nv_bins
    total_int = np.sum(f_fluid) * dx * dv
    if total_int > 0:
        f_fluid /= total_int

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    X_grid, V_grid = np.meshgrid(x_centers, v_centers)

    return X_grid, V_grid, f_fluid


def compute_fluid_moments_2d(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    sigma: float = 1.2,
) -> Dict[str, np.ndarray]:
    """Compute macroscopic fluid moments from 2D particle ensemble.

    Calculates:
    - n(x, y): Fluid number density
    - ux(x, y), uy(x, y): Bulk fluid velocity vector field
    - T(x, y): Kinetic temperature (velocity dispersion)
    - vorticity(x, y): Fluid vorticity omega_z = duy/dx - dux/dy

    Returns
    -------
    moments : Dict[str, np.ndarray]
        Dictionary containing 'density', 'ux', 'uy', 'temperature', 'vorticity'.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    vx = np.asarray(vx).ravel()
    vy = np.asarray(vy).ravel()

    dx = Lx / Nx
    dy = Ly / Ny

    # Normalized indices
    xi = np.mod(np.floor(x / dx).astype(np.int64), Nx)
    yj = np.mod(np.floor(y / dy).astype(np.int64), Ny)
    k = yj * Nx + xi
    total_cells = Nx * Ny

    # Moment 0: Number of particles per cell
    count_flat = np.bincount(k, minlength=total_cells).astype(np.float64)
    # Moment 1: Momentum
    px_flat = np.bincount(k, weights=vx, minlength=total_cells)
    py_flat = np.bincount(k, weights=vy, minlength=total_cells)
    # Moment 2: Kinetic energy sum
    vsq_flat = np.bincount(k, weights=(vx**2 + vy**2), minlength=total_cells)

    count = count_flat.reshape((Ny, Nx))
    px = px_flat.reshape((Ny, Nx))
    py = py_flat.reshape((Ny, Nx))
    vsq = vsq_flat.reshape((Ny, Nx))

    # Apply periodic Gaussian smoothing to suppress macroparticle discrete noise
    count_s = ndi.gaussian_filter(count, sigma=sigma, mode="wrap")
    px_s = ndi.gaussian_filter(px, sigma=sigma, mode="wrap")
    py_s = ndi.gaussian_filter(py, sigma=sigma, mode="wrap")
    vsq_s = ndi.gaussian_filter(vsq, sigma=sigma, mode="wrap")

    # Safe division for bulk velocity
    mask = count_s > 1e-6
    ux = np.zeros_like(count_s)
    uy = np.zeros_like(count_s)
    temp = np.zeros_like(count_s)

    ux[mask] = px_s[mask] / count_s[mask]
    uy[mask] = py_s[mask] / count_s[mask]

    # Thermal dispersion T = <v^2> - |u|^2
    mean_vsq = np.zeros_like(count_s)
    mean_vsq[mask] = vsq_s[mask] / count_s[mask]
    u_mag_sq = ux**2 + uy**2
    temp = np.maximum(0.5 * (mean_vsq - u_mag_sq), 0.0)

    # Vorticity: curl(u)_z = duy/dx - dux/dy with periodic central differencing
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dx)
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dy)
    vorticity = duy_dx - dux_dy

    density = count_s / (dx * dy)

    return {
        "density": density,
        "ux": ux,
        "uy": uy,
        "temperature": temp,
        "vorticity": vorticity,
    }


def render_phase_fluid_frame(
    ax: plt.Axes,
    x: np.ndarray,
    vx: np.ndarray,
    boxsize: float,
    v_lim: float = 6.0,
    E_field: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    cmap: str = "magma",
    title: str = "Phase-Space Fluid Distribution",
):
    """Render a continuous phase-space fluid density with overlaid streamline contours."""
    X_grid, V_grid, f_fluid = reconstruct_phase_fluid(x, vx, boxsize, v_lim=v_lim)

    im = ax.imshow(
        f_fluid,
        origin="lower",
        extent=[0, boxsize, -v_lim, v_lim],
        cmap=cmap,
        aspect="auto",
        interpolation="bicubic",
    )

    # Add subtle density contour levels to emphasize fluid vortex ripples
    max_f = np.max(f_fluid)
    if max_f > 0:
        levels = np.linspace(0.15 * max_f, 0.95 * max_f, 5)
        ax.contour(
            X_grid,
            V_grid,
            f_fluid,
            levels=levels,
            colors="#ffffff",
            alpha=0.25,
            linewidths=0.7,
        )

    # If electric field is given, overlay Hamiltonian phase-space streamline trajectories!
    # In phase space: dx/dt = vx, dvx/dt = q*E/m
    if E_field is not None and grid_x is not None:
        E_interp = np.interp(X_grid, grid_x, E_field)
        U_phase = V_grid
        V_phase = -1.0 * E_interp  # q/m = -1 for electrons
        # Subsample for clean streamplot
        sub_x = slice(None, None, 14)
        sub_v = slice(None, None, 14)
        ax.streamplot(
            X_grid[sub_v, sub_x],
            V_grid[sub_v, sub_x],
            U_phase[sub_v, sub_x],
            V_phase[sub_v, sub_x],
            color="#38bdf8",
            density=0.8,
            linewidth=0.6,
            arrowsize=0.6,
        )

    ax.set_xlim(0, boxsize)
    ax.set_ylim(-v_lim, v_lim)
    ax.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax.set_ylabel(r"Velocity $v_x / v_{th}$")
    ax.set_title(title, fontweight="bold")
    return im
