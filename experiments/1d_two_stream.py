"""
1D-3V Two-Stream Instability Benchmark Simulation
=================================================
Simulates the nonlinear evolution of two counter-streaming electron beams
in a neutralizing uniform background.

Features:
- Solves Vlasov-Poisson dynamics with symplectic Boris leapfrog & CIC.
- Compares measured electrostatic field growth rate against kinetic linear dispersion theory.
- Visualizes phase-space vortex formation, Fourier mode cascades, and energy conservation.

Run:
    python experiments/1d_two_stream.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_two_stream_instability():
    apply_epic_style()
    Nx = 400                    # Grid cells
    boxsize = 45.0              # Domain length L
    N_particles = 120000        # Total particle count (60,000 per beam)
    dt = 0.1                    # Time step (omega_pe * dt = 0.1, stable!)
    t_end = 50.0                # Final time
    v_beam = 3.0                # Drift velocity
    v_thermal = 0.5             # Thermal velocity spread
    n0 = 1.0                    # Total background electron density

    weight = (n0 * boxsize) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    print("==========================================================")
    print("      ePic 1D-3V TWO-STREAM INSTABILITY BENCHMARK        ")
    print("==========================================================")
    print(f"Grid: Nx = {Nx}, L = {boxsize}, dx = {boxsize/Nx:.4f}")
    print(f"Particles: N = {N_particles} ({N_particles//2} per beam)")
    print(f"Time: dt = {dt}, t_end = {t_end} ({int(t_end/dt)} steps)")

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    Nh = N_particles // 2
    np.random.seed(42)

    # Beam 1 (Right-drifting)
    pos1 = np.random.uniform(0.0, boxsize, Nh)
    vel1 = np.zeros((Nh, 3))
    vel1[:, 0] = np.random.normal(v_beam, v_thermal, Nh)

    # Beam 2 (Left-drifting)
    pos2 = np.random.uniform(0.0, boxsize, Nh)
    vel2 = np.zeros((Nh, 3))
    vel2[:, 0] = np.random.normal(-v_beam, v_thermal, Nh)

    solver.add_species("beam_right", q=q_macro, m=m_macro, pos=pos1, vel=vel1)
    solver.add_species("beam_left", q=q_macro, m=m_macro, pos=pos2, vel=vel2)
    solver.initialize()

    snapshots = {}
    target_times = [0.0, 15.0, 25.0, 45.0]

    # Modal history for modes m=1, 2, 3, 4
    mode_hist = {1: [], 2: [], 3: [], 4: []}

    def record_snapshot(s):
        # Record spatial Fourier modes of E
        E_fft = np.fft.rfft(s.E)
        for m in [1, 2, 3, 4]:
            if m < len(E_fft):
                mode_hist[m].append(np.abs(E_fft[m]) ** 2)

        for tt in target_times:
            if abs(s.time - tt) < s.dt / 2 and tt not in snapshots:
                p_r = s.species[0].pos[:, 0].copy()
                v_r = s.species[0].vel[:, 0].copy()
                p_l = s.species[1].pos[:, 0].copy()
                v_l = s.species[1].vel[:, 0].copy()
                snapshots[tt] = (p_r, v_r, p_l, v_l)
                print(f"  --> Captured phase-space snapshot at t = {s.time:.1f}")

    record_snapshot(solver)

    print("\nEvolving Vlasov-Poisson dynamics...")
    solver.run(t_end=t_end, callback=record_snapshot, callback_interval=1)

    print("\nSimulation complete! Generating creative analysis dashboards...")
    os.makedirs("docs/images", exist_ok=True)

    # -------------------------------------------------------------
    # 1. CREATIVE PHASE-SPACE EVOLUTION DASHBOARD (Dark Mode)
    # -------------------------------------------------------------
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=140, facecolor="#090d16")
    axes = axes.ravel()

    for idx, tt in enumerate(target_times):
        ax = axes[idx]
        ax.set_facecolor("#090d16")
        if tt in snapshots:
            p_r, v_r, p_l, v_l = snapshots[tt]
            sub = slice(None, None, 2)
            # Use high-contrast glowing neon cyan & crimson
            ax.scatter(p_r[sub], v_r[sub], s=0.4, color="#38bdf8", alpha=0.45, label="Beam 1 (+v)")
            ax.scatter(p_l[sub], v_l[sub], s=0.4, color="#f43f5e", alpha=0.45, label="Beam 2 (-v)")

        ax.set_xlim(0, boxsize)
        ax.set_ylim(-7.5, 7.5)
        ax.set_xlabel(r"Spatial Coordinate $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
        ax.set_ylabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
        ax.set_title(rf"Phase Space at $\omega_{{pe}} t = {tt:.1f}$", fontsize=13, fontweight="bold", color="#f8fafc")
        if idx == 0:
            ax.legend(loc="upper right", markerscale=8, framealpha=0.6, facecolor="#1e293b", edgecolor="none")
        ax.grid(True, linestyle="--", alpha=0.2, color="#475569")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Two-Stream Kinetic Phase-Space Vortex Roll-Up",
        subtitle="Nonlinear BGK Solitary Wave Coalescence and Phase Mixing",
        hspace=0.32,
        wspace=0.24,
    )
    save_epic_plot(fig, "docs/images/two_stream_phase_space.png")

    # -------------------------------------------------------------
    # 2. CREATIVE MULTI-PANEL ENERGY & MODAL GROWTH DASHBOARD
    # -------------------------------------------------------------
    time_arr = np.array(solver.history["time"])
    e_kin = np.array(solver.history["E_kin"])
    e_field = np.array(solver.history["E_field"])
    e_tot = np.array(solver.history["E_total"])

    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor=EPIC_COLORS["bg_dark"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    # Panel 1: Energy Partition & Exact Second-Order Conservation
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    ax1.plot(time_arr, e_kin / e_tot[0], label="Kinetic Energy ($E_{kin}$)", color="#38bdf8", lw=2.2)
    ax1.plot(time_arr, e_tot / e_tot[0], label="Total Symplectic Energy ($E_{tot}$)", color="#f8fafc", linestyle="--", lw=1.8)
    ax1.plot(time_arr, e_field / e_tot[0], label="Electrostatic Field ($E_{field}$)", color="#f43f5e", lw=2.2)

    # Inset showing relative energy drift
    rel_drift_arr = np.abs(e_tot - e_tot[0]) / e_tot[0]
    ax1_ins = ax1.inset_axes([0.48, 0.28, 0.48, 0.35], facecolor="#1e293b")
    ax1_ins.plot(time_arr, rel_drift_arr, color="#34d399", lw=1.5)
    ax1_ins.set_yscale("log")
    ax1_ins.set_title(rf"Relative Drift $\Delta E / E_0 < {np.max(rel_drift_arr):.1e}$", fontsize=9, color="#34d399")
    ax1_ins.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1_ins.tick_params(colors="#f8fafc", labelsize=8)

    ax1.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Normalized Energy $\mathcal{E} / \mathcal{E}_0$", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) Symplectic Energy Partition & Leapfrog Conservation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="lower left", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # Panel 2: Field Energy & Linear Dispersion Theory Fit
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    nonzero_field = np.maximum(e_field, 1e-12)
    ax2.semilogy(time_arr, nonzero_field, color="#f43f5e", lw=2.2, label=r"Measured Field Energy $\mathcal{E}_{field}(t)$")

    mask = (time_arr >= 10.0) & (time_arr <= 22.0)
    fit = np.polyfit(time_arr[mask], np.log(nonzero_field[mask]), 1)
    gamma_measured = fit[0] / 2.0
    ax2.plot(
        time_arr[mask],
        np.exp(fit[1] + fit[0] * time_arr[mask]),
        "--",
        color="#fbbf24",
        lw=2.2,
        label=rf"Linear Fit: $\gamma \approx {gamma_measured:.3f} \, \omega_{{pe}}$ (Theory $\approx 0.35$)",
    )
    ax2.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Field Energy $\mathcal{E}_{field}$ (Log Scale)", fontsize=11, color="#f8fafc")
    ax2.set_title(r"(b) Exponential Linear Growth & Nonlinear Saturation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax2.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax2.legend(loc="lower right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # Panel 3: Modal Fourier Harmonic Decomposition |E_m|^2(t)
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    m_colors = ["#38bdf8", "#34d399", "#fbbf24", "#a855f7"]
    for m, col in zip([1, 2, 3, 4], m_colors):
        if len(mode_hist[m]) == len(time_arr):
            ax3.semilogy(time_arr, np.maximum(mode_hist[m], 1e-12), color=col, lw=2.0, label=rf"Mode $m={m}$ ($k = {m*2*np.pi/boxsize:.2f}$)")

    ax3.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Mode Power $|E_m|^2$", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) Spatial Fourier Harmonic Cascades & Mode Coupling", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="lower right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # Panel 4: Spatially Averaged Velocity Distribution Evolution f(v)
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    v_bins = np.linspace(-7.0, 7.0, 80)
    p_r0, v_r0, p_l0, v_l0 = snapshots[0.0]
    p_rF, v_rF, p_lF, v_lF = snapshots[45.0]
    v_init_all = np.concatenate([v_r0, v_l0])
    v_final_all = np.concatenate([v_rF, v_lF])

    ax4.hist(v_init_all, bins=v_bins, density=True, histtype="step", color="#38bdf8", lw=2.4, label=r"Initial Beams ($t=0$)")
    ax4.hist(v_final_all, bins=v_bins, density=True, histtype="stepfilled", color="#f43f5e", alpha=0.35, label=r"Thermalized Plateau ($t=45$)")
    ax4.set_xlabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Distribution Function $f(v_x)$", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) Velocity Space Thermalization & Vortex Trapping Plateau", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Two-Stream Instability Diagnostic Dashboard",
        subtitle="Energy Conservation, Kinetic Growth Rate Verification, and Harmonic Cascades",
    )
    save_epic_plot(fig, "docs/images/two_stream_energy.png")

    rel_drift = abs(e_tot[-1] - e_tot[0]) / e_tot[0]
    print(f"\nFinal Relative Energy Conservation Drift: {rel_drift:.2e}")
    print(f"Measured Exponential Growth Rate: gamma = {gamma_measured:.4f} omega_pe")
    print("==========================================================")


if __name__ == "__main__":
    run_two_stream_instability()
