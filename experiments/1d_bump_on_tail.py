"""
1D-3V Bump-on-Tail Instability & Quasilinear Plateau Formation
=============================================================
Simulates a weak high-energy electron beam traversing a warm background plasma.
Because df/dv > 0 at resonant velocities, waves grow via inverse Landau damping,
scattering electrons and flattening the distribution into a stable quasilinear plateau.

Run:
    python experiments/1d_bump_on_tail.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_bump_on_tail():
    apply_epic_style()
    Nx = 256
    boxsize = 60.0
    N_core = 110000
    N_beam = 30000
    dt = 0.1
    t_end = 60.0

    v_th_core = 1.0
    v_beam = 4.5
    v_th_beam = 0.6

    n_core = 0.88
    n_beam = 0.12
    total_n = n_core + n_beam

    w_core = (n_core * boxsize) / N_core
    w_beam = (n_beam * boxsize) / N_beam

    print("==========================================================")
    print("      ePic 1D-3V BUMP-ON-TAIL INSTABILITY & PLATEAU      ")
    print("==========================================================")
    print(f"Domain: L = {boxsize}, Nx = {Nx}, Particles: {N_core + N_beam}")
    print(f"Core: n={n_core}, v_th={v_th_core} | Beam: n={n_beam}, v_b={v_beam}, v_th={v_th_beam}")

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    np.random.seed(42)
    # Core plasma
    x_c = np.random.uniform(0.0, boxsize, N_core)
    v_c = np.zeros((N_core, 3))
    v_c[:, 0] = np.random.normal(0.0, v_th_core, N_core)

    # Beam plasma
    x_b = np.random.uniform(0.0, boxsize, N_beam)
    v_b = np.zeros((N_beam, 3))
    v_b[:, 0] = np.random.normal(v_beam, v_th_beam, N_beam)

    solver.add_species("core_electrons", q=-w_core, m=w_core, pos=x_c, vel=v_c)
    solver.add_species("beam_electrons", q=-w_beam, m=w_beam, pos=x_b, vel=v_b)
    solver.initialize()

    # Capture velocity distributions at multiple epochs
    v_epochs = {}
    target_epochs = [0.0, 15.0, 30.0, 60.0]

    def record_epoch(s):
        for te in target_epochs:
            if abs(s.time - te) < s.dt / 2 and te not in v_epochs:
                v_all = np.concatenate([s.species[0].vel[:, 0], s.species[1].vel[:, 0]])
                v_epochs[te] = v_all.copy()

    record_epoch(solver)

    print("\nEvolving quasilinear wave-particle interaction...")
    solver.run(t_end=t_end, callback=record_epoch, callback_interval=1)

    print("\nGenerating Creative Bump-on-Tail Dashboard...")
    os.makedirs("docs/images", exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor=EPIC_COLORS["bg_dark"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    # -------------------------------------------------------------
    # Panel 1: Multi-Epoch Quasilinear Plateau Relaxation
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    v_bins = np.linspace(-3.5, 7.5, 90)
    epoch_colors = ["#38bdf8", "#34d399", "#fbbf24", "#f43f5e"]

    for te, col in zip(target_epochs, epoch_colors):
        if te in v_epochs:
            if te == 0.0:
                ax1.hist(v_epochs[te], bins=v_bins, density=True, histtype="step", color=col, lw=2.4, label=rf"Initial $f(v)$ ($t=0$) [$\partial f/\partial v > 0$]")
            elif te == 60.0:
                ax1.hist(v_epochs[te], bins=v_bins, density=True, histtype="stepfilled", color=col, alpha=0.35, label=rf"Quasilinear Plateau ($t={te:.0f}$) [$\partial f/\partial v = 0$]")
                ax1.hist(v_epochs[te], bins=v_bins, density=True, histtype="step", color=col, lw=2.0)
            else:
                ax1.hist(v_epochs[te], bins=v_bins, density=True, histtype="step", color=col, lw=1.8, linestyle="--", label=rf"$t={te:.0f}\,\omega_{{pe}}^{{-1}}$")

    # Annotate plateau region
    ax1.axvspan(3.2, 5.5, color="#fbbf24", alpha=0.12, label="Resonant Diffusion Plateau")
    ax1.set_xlim(-3.5, 7.5)
    ax1.set_ylim(0.0, 0.44)
    ax1.set_xlabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Distribution Function $f(v_x)$", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) Quasilinear Relaxation & Plateau Formation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 2: Phase-Space Vortex Trapping at Saturation
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    sp_c_x = solver.species[0].pos[:, 0]
    sp_c_v = solver.species[0].vel[:, 0]
    sp_b_x = solver.species[1].pos[:, 0]
    sp_b_v = solver.species[1].vel[:, 0]

    sub_c = slice(None, None, 5)
    sub_b = slice(None, None, 2)

    ax2.scatter(sp_c_x[sub_c], sp_c_v[sub_c], s=0.4, color="#38bdf8", alpha=0.3, label="Thermal Core")
    ax2.scatter(sp_b_x[sub_b], sp_b_v[sub_b], s=0.8, color="#f43f5e", alpha=0.5, label="Trapped Resonant Beam")
    ax2.axhline(v_beam, color="#fbbf24", linestyle=":", lw=1.8, label=r"Initial Beam Velocity $v_b = 4.5\,v_{th}$")

    ax2.set_xlim(0, boxsize)
    ax2.set_ylim(-4.0, 8.5)
    ax2.set_xlabel(r"Spatial Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
    ax2.set_title(rf"(b) Kinetic Phase Space Trapping Island ($t = {t_end}\ \omega_{{pe}}^{{-1}}$)", fontsize=13, fontweight="bold", color="#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax2.legend(loc="upper left", markerscale=6, framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 3: Wave Field Energy Growth & Quasilinear Saturation
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    t_arr = np.array(solver.history["time"])
    e_field = np.array(solver.history["E_field"])
    ax3.semilogy(t_arr, np.maximum(e_field, 1e-12), color="#34d399", lw=2.2, label=r"Measured Field Energy $\mathcal{E}_{field}(t)$")

    # Linear inverse Landau growth phase
    mask = (t_arr >= 5.0) & (t_arr <= 22.0)
    fit = np.polyfit(t_arr[mask], np.log(np.maximum(e_field[mask], 1e-12)), 1)
    gamma_measured = fit[0] / 2.0
    ax3.plot(
        t_arr[mask],
        np.exp(fit[1] + fit[0] * t_arr[mask]),
        "--",
        color="#fbbf24",
        lw=2.0,
        label=rf"Inverse Landau Growth: $\gamma \approx {gamma_measured:.3f} \, \omega_{{pe}}$",
    )

    ax3.set_xlim(0, t_end)
    ax3.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Electrostatic Wave Energy $\mathcal{E}_{field}$", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) Inverse Landau Wave Growth & Nonlinear Saturation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="lower right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 4: Energy Conservation & Beam Deceleration
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    e_kin = np.array(solver.history["E_kin"])
    e_tot = np.array(solver.history["E_total"])

    delta_kin = e_kin - e_kin[0]
    ax4.plot(t_arr, -delta_kin, color="#f43f5e", lw=2.2, label=r"Kinetic Energy Lost by Beam $-\Delta \mathcal{E}_{kin}$")
    ax4.plot(t_arr, e_field - e_field[0], color="#38bdf8", lw=2.0, linestyle="--", label=r"Wave Field Energy Gained $\Delta \mathcal{E}_{field}$")

    ax4.set_xlim(0, t_end)
    ax4.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Energy Conversion $\Delta \mathcal{E}$", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) Kinetic-to-Wave Energy Transfer Dynamics", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper left", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Bump-on-Tail Quasilinear Relaxation Benchmark",
        subtitle="Inverse Landau Damping, Resonant Diffusion, and Flattened Velocity Plateau Formation",
    )
    save_epic_plot(fig, "docs/images/bump_on_tail_plateau.png")
    print("==========================================================")


if __name__ == "__main__":
    run_bump_on_tail()
