"""
2D-3V Kelvin-Helmholtz Instability & Kinetic Vortex Roll-up
===========================================================
Simulates the nonlinear kinetic evolution of a sheared plasma flow:
    v_x(y) = v_0 * tanh((y - Ly/2) / L_s)

The velocity shear excites the Kelvin-Helmholtz instability (KHI),
causing the shear layer to roll up into macroscopic, swirling fluid-like
cat's-eye vortices and secondary kinetic filamentation eddies.

Demonstrates:
- Fluid-moment reconstruction: Vorticity omega_z = (curl u)_z, bulk flow u(x, y)
- Kinetic temperature heating within vortex cores
- Cinema-grade fluid visualization with dark obsidian styling.

Run:
    python experiments/2d_kelvin_helmholtz.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic2d import PIC2DSolver
from epic.diagnostics import (
    apply_epic_style,
    format_epic_figure,
    save_epic_plot,
    compute_fluid_moments_2d,
    EPIC_COLORS,
)


def run_kelvin_helmholtz():
    apply_epic_style()

    Nx = 128
    Ny = 128
    Lx = 30.0
    Ly = 30.0
    N_particles = 120000
    dt = 0.08
    t_end = 35.0

    v0 = 2.0                    # Shear flow velocity
    L_s = 2.5                   # Shear layer thickness
    v_th = 0.35                 # Thermal velocity spread
    y_center = Ly / 2.0
    delta_vy = 0.15 * v0        # Seed transverse perturbation

    print("==========================================================")
    print("   ePic 2D-3V KELVIN-HELMHOLTZ INSTABILITY & VORTEX       ")
    print("==========================================================")
    print(f"Domain: {Lx} x {Ly}, Grid: {Nx} x {Ny}, Shear width: {L_s}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    weight = (1.0 * Lx * Ly) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC2DSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dt=dt)

    np.random.seed(42)
    x = np.random.uniform(0.0, Lx, N_particles)
    y = np.random.uniform(0.0, Ly, N_particles)

    # Sheared velocity profile: vx = v0 * tanh((y - yc) / Ls)
    tanh_y = np.tanh((y - y_center) / L_s)
    vx = v0 * tanh_y + np.random.normal(0.0, v_th, N_particles)

    # Transverse seed perturbation localized at shear layer
    pert_env = np.exp(-((y - y_center) / (1.8 * L_s)) ** 2)
    vy = delta_vy * np.sin(2.0 * np.pi * x / Lx) * pert_env + np.random.normal(0.0, v_th, N_particles)
    vz = np.random.normal(0.0, v_th, N_particles)

    vel = np.column_stack((vx, vy, vz))

    solver.add_species("sheared_plasma", q=q_macro, m=m_macro, pos_x=x, pos_y=y, vel=vel)
    solver.initialize()

    print("\nEvolving kinetic shear flow dynamics...")
    solver.run(t_end=t_end)

    print("\nReconstructing macroscopic fluid fields (Vorticity, Flow, Pressure)...")
    sp = solver.species[0]
    moments = compute_fluid_moments_2d(
        sp.x, sp.y, sp.vel[:, 0], sp.vel[:, 1],
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, sigma=1.4,
    )

    density = moments["density"]
    ux = moments["ux"]
    uy = moments["uy"]
    vorticity = moments["vorticity"]
    temp = moments["temperature"]

    # -------------------------------------------------------------
    # 4-PANEL PUBLICATION DASHBOARD: KINETIC-TO-FLUID VISUALIZATION
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12), dpi=140)
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28, top=0.88, bottom=0.08, left=0.07, right=0.95)

    gx = np.linspace(0.0, Lx, Nx)
    gy = np.linspace(0.0, Ly, Ny)

    # Panel 1: Fluid Vorticity Field & Swirling Streamlines
    ax1 = fig.add_subplot(gs[0, 0])
    vort_lim = np.percentile(np.abs(vorticity), 98)
    im1 = ax1.imshow(
        vorticity,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
        vmin=-vort_lim,
        vmax=vort_lim,
    )
    # Streamlines of bulk velocity u(x, y)
    sub_x = slice(None, None, 2)
    sub_y = slice(None, None, 2)
    ax1.streamplot(
        gx[sub_x],
        gy[sub_y],
        ux[sub_y, sub_x],
        uy[sub_y, sub_x],
        color="#ffffff",
        density=1.3,
        linewidth=0.8,
        arrowsize=0.8,
    )
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Fluid Vorticity $\omega_z = (\nabla \times \mathbf{u})_z$", color=EPIC_COLORS["text"], labelpad=8)
    ax1.set_xlabel(r"Streamwise Coordinate $x$ ($c/\omega_{pe}$)")
    ax1.set_ylabel(r"Transverse Coordinate $y$ ($c/\omega_{pe}$)")
    ax1.set_title(r"(a) Fluid Vorticity $\omega_z$ & Cat's-Eye Streamlines", fontweight="bold")

    # Panel 2: Continuous Charge Density Fluid Field
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        density,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="inferno",
        aspect="auto",
    )
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Plasma Density $n(x, y)$", color=EPIC_COLORS["text"], labelpad=8)
    ax2.set_xlabel(r"$x$ ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"$y$ ($c/\omega_{pe}$)")
    ax2.set_title(r"(b) Continuous Plasma Fluid Density & Vortex Core", fontweight="bold")

    # Panel 3: Kinetic Temperature / Pressure Heating
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        temp,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r"Kinetic Temperature $T(x, y) = \frac{1}{2} \langle (\mathbf{v}-\mathbf{u})^2 \rangle$", color=EPIC_COLORS["text"], labelpad=8)
    ax3.set_xlabel(r"$x$ ($c/\omega_{pe}$)")
    ax3.set_ylabel(r"$y$ ($c/\omega_{pe}$)")
    ax3.set_title(r"(c) Non-Thermal Kinetic Heating in Vortex Shear Layer", fontweight="bold")

    # Panel 4: Energy Partition and Saturation
    ax4 = fig.add_subplot(gs[1, 1])
    t_arr = np.array(solver.history["time"])
    e_kin = np.array(solver.history["E_kin"])
    e_field = np.array(solver.history["E_field"])
    e_tot = np.array(solver.history["E_total"])

    ax4.plot(t_arr, e_kin / e_tot[0], color=EPIC_COLORS["cyan"], lw=2.2, label=r"Kinetic Energy $\mathcal{E}_{kin}$")
    ax4.plot(t_arr, e_tot / e_tot[0], color=EPIC_COLORS["gold"], linestyle="--", lw=1.8, label=r"Total Energy $\mathcal{E}_{tot}$")
    ax4.plot(t_arr, (e_field / e_tot[0]) * 100.0, color=EPIC_COLORS["crimson"], lw=2.0, label=r"Field Energy $\mathcal{E}_{field} \times 100$")

    ax4.set_xlabel(r"Time ($\omega_{pe} t$)")
    ax4.set_ylabel(r"Normalized Energy $\mathcal{E} / \mathcal{E}_0$")
    ax4.set_title(r"(d) Energy Conservation & Shear Dissipation", fontweight="bold")
    ax4.legend(loc="lower left")

    format_epic_figure(
        fig,
        title="ePic 2D-3V Kelvin-Helmholtz Instability Benchmark",
        subtitle="Nonlinear Vortex Roll-up, Cat's-Eye Eddies & Kinetic-to-Fluid Moment Transformation",
    )

    output_path = "docs/images/kelvin_helmholtz_vortex.png"
    save_epic_plot(fig, output_path)

    drift = abs(e_tot[-1] - e_tot[0]) / e_tot[0]
    print(f"\nFinal Relative Energy Drift: {drift:.2e}")
    print(f"Max Vorticity Generated: |omega|_max = {np.max(np.abs(vorticity)):.3f}")
    print("==========================================================")


if __name__ == "__main__":
    run_kelvin_helmholtz()
