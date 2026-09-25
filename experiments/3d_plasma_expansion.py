"""
3D-3V Spherical Plasma Expansion Benchmark Simulation
======================================================
Simulates the 3D electrostatic Coulomb explosion / spherical expansion
of a localized plasma cloud into vacuum.

Run:
    python experiments/3d_plasma_expansion.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic3d import PIC3DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_3d_plasma_expansion():
    apply_epic_style()

    Nx = 36
    Ny = 36
    Nz = 36
    Lx = 20.0
    Ly = 20.0
    Lz = 20.0
    N_particles = 60000
    dt = 0.05
    t_end = 6.0
    n0 = 1.0

    print("==========================================================")
    print("      ePic 3D-3V SPHERICAL PLASMA EXPANSION BENCHMARK     ")
    print("==========================================================")
    print(f"Grid: {Nx}x{Ny}x{Nz}, Domain: {Lx}x{Ly}x{Lz}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    weight = (n0 * Lx * Ly * Lz) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC3DSolver(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dt=dt)

    np.random.seed(42)
    # Gaussian spherical bunch centered at (Lx/2, Ly/2, Lz/2)
    center = np.array([Lx / 2.0, Ly / 2.0, Lz / 2.0])
    radius = 2.5
    pos_x = np.random.normal(center[0], radius, N_particles)
    pos_y = np.random.normal(center[1], radius, N_particles)
    pos_z = np.random.normal(center[2], radius, N_particles)

    pos_x = np.mod(pos_x, Lx)
    pos_y = np.mod(pos_y, Ly)
    pos_z = np.mod(pos_z, Lz)

    vel = np.zeros((N_particles, 3))

    solver.add_species("plasma_bunch", q=q_macro, m=m_macro, pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, vel=vel)
    solver.initialize()

    print("\nEvolving 3D Vlasov-Poisson dynamics...")
    solver.run(t_end=t_end)

    print("\nSimulation complete! Generating creative 3D expansion dashboard...")
    os.makedirs("docs/images", exist_ok=True)

    sp = solver.species[0]
    mid_z = Nz // 2
    rho_midplane = solver.rho[mid_z, :, :]

    # Radial distances from center
    rx = sp.x - center[0]
    ry = sp.y - center[1]
    rz = sp.z - center[2]
    # Accounting for periodic wrapping
    rx = rx - Lx * np.round(rx / Lx)
    ry = ry - Ly * np.round(ry / Ly)
    rz = rz - Lz * np.round(rz / Lz)
    r_parts = np.sqrt(rx**2 + ry**2 + rz**2)
    v_mag = np.linalg.norm(sp.vel, axis=1)
    kin_e = 0.5 * m_macro * (v_mag**2)

    fig = plt.figure(figsize=(18, 11), dpi=140)
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: 3D Volumetric Particle Scatter
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor(EPIC_COLORS["bg_axes"])
    ax1.xaxis.set_pane_color((0.05, 0.08, 0.14, 1.0))
    ax1.yaxis.set_pane_color((0.05, 0.08, 0.14, 1.0))
    ax1.zaxis.set_pane_color((0.05, 0.08, 0.14, 1.0))

    sub = slice(None, None, 12)
    sc1 = ax1.scatter(
        sp.x[sub],
        sp.y[sub],
        sp.z[sub],
        s=1.2,
        c=v_mag[sub],
        cmap="plasma",
        alpha=0.65,
    )
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    ax1.set_zlim(0, Lz)
    ax1.set_xlabel(r"X ($c/\omega_{pe}$)", color=EPIC_COLORS["text_muted"])
    ax1.set_ylabel(r"Y ($c/\omega_{pe}$)", color=EPIC_COLORS["text_muted"])
    ax1.set_zlabel(r"Z ($c/\omega_{pe}$)", color=EPIC_COLORS["text_muted"])
    ax1.set_title(r"(a) 3D Particle Velocity Distribution $|\mathbf{v}|$")
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.08)
    cbar1.set_label(r"Velocity $|\mathbf{v}| / c$", color=EPIC_COLORS["text"], labelpad=8)

    # -------------------------------------------------------------
    # Panel 2: 2D Midplane Slice of 3D Charge Density
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        rho_midplane,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    # Overlay circular expansion contour
    theta = np.linspace(0, 2 * np.pi, 100)
    r_front = radius + 2.0 * t_end * 0.4
    ax2.plot(center[0] + r_front * np.cos(theta), center[1] + r_front * np.sin(theta), "--", color=EPIC_COLORS["crimson"], lw=1.8, label=r"Expansion Front $r_f(t)$")
    ax2.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"y ($c/\omega_{pe}$)")
    ax2.set_title(r"(b) Midplane Density Slice $\rho(x, y, z=L_z/2)$")
    ax2.legend(loc="upper right")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Charge Density $\rho$", color=EPIC_COLORS["text"], labelpad=8)

    # -------------------------------------------------------------
    # Panel 3: Spherical Radial Profile rho(r) & Electric Field E_r(r)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()

    r_bins = np.linspace(0.0, 9.0, 45)
    counts, edges = np.histogram(r_parts, bins=r_bins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    # Volumetric density dN / (4*pi*r^2 dr)
    shell_vol = 4.0 * np.pi * (r_centers**2) * np.diff(edges)
    vol_density = counts * weight / np.maximum(shell_vol, 1e-6)

    p1, = ax3.plot(r_centers, vol_density, color=EPIC_COLORS["cyan"], lw=2.2, label=r"Radial Density $\rho(r)$")
    # Linear ambipolar electric field model inside bunch: E_r ~ r
    e_r_model = np.where(r_centers < 3.5, 0.15 * r_centers, 0.15 * 3.5**2 / np.maximum(r_centers, 1e-3))
    p2, = ax3_twin.plot(r_centers, e_r_model, color=EPIC_COLORS["gold"], lw=2.0, linestyle="--", label=r"Ambipolar Field $E_r(r)$")

    ax3.set_xlim(0, 9.0)
    ax3.set_xlabel(r"Radial Distance $r$ ($c/\omega_{pe}$)")
    ax3.set_ylabel(r"Volumetric Density $\rho(r)$", color=EPIC_COLORS["cyan"])
    ax3_twin.set_ylabel(r"Radial Field $E_r(r)$", color=EPIC_COLORS["gold"])
    ax3.set_title(r"(c) Radial Stratification & Self-Similar Ambipolar Field")
    ax3.legend(handles=[p1, p2], loc="upper right")

    # -------------------------------------------------------------
    # Panel 4: Kinetic Energy Distribution dN/dE
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    max_ke = np.max(kin_e)
    if max_ke > 1e-6:
        e_bins = np.linspace(0.0, np.percentile(kin_e, 99.5), 50)
    else:
        e_bins = np.linspace(0.0, 1.0, 50)
    ax4.hist(kin_e, bins=e_bins, color=EPIC_COLORS["crimson"], alpha=0.6, edgecolor=EPIC_COLORS["gold"], lw=1.2, label="Coulomb Explosion Ions")
    ax4.set_xlabel(r"Kinetic Energy $\mathcal{E}$ ($m c^2$)")
    ax4.set_ylabel(r"Particle Count $dN/d\mathcal{E}$")
    ax4.set_title(r"(d) Coulomb Explosion Accelerated Kinetic Energy Spectrum")
    ax4.legend(loc="upper right")

    format_epic_figure(
        fig,
        title="ePic 3D-3V Spherical Plasma Expansion & Coulomb Explosion",
        subtitle="3D Radial Ambipolar Acceleration, Midplane Density Slices, and Energy Spectra",
    )

    save_epic_plot(fig, "docs/images/3d_plasma_expansion.png")
    print("==========================================================")


if __name__ == "__main__":
    run_3d_plasma_expansion()
