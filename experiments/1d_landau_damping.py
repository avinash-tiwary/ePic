"""
1D-3V Collisionless Landau Damping Benchmark Simulation
======================================================
Simulates the collisionless damping of an electrostatic Langmuir wave in a
thermal Maxwellian plasma, comparing numerical decay against linear Landau theory.

Theory:
    Linear Landau damping rate:
        gamma_L = - sqrt(pi/8) * (omega_pe / (k * lambda_D)^3) * exp(-1/(2*(k*lambda_D)^2) - 1.5)
    Real frequency (Bohm-Gross):
        omega_r = sqrt(omega_pe^2 + 3 * (k * lambda_D)^2 * omega_pe^2)

Run:
    python experiments/1d_landau_damping.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_landau_damping():
    apply_epic_style()
    Nx = 256
    k = 0.5
    boxsize = 2.0 * np.pi / k    # L = 4*pi ~ 12.566
    N_particles = 160000        # High particle count to suppress thermal noise
    dt = 0.05
    t_end = 30.0
    v_th = 1.0                  # Thermal velocity -> lambda_D = v_th / omega_pe = 1.0
    n0 = 1.0
    alpha = 0.08                # Perturbation amplitude (linear regime)

    print("==========================================================")
    print("        ePic 1D-3V COLLISIONLESS LANDAU DAMPING          ")
    print("==========================================================")
    print(f"Domain: L = {boxsize:.4f}, k = {k}, k*lambda_D = {k*v_th:.2f}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    # Analytic Landau damping rate
    k_ld = k * v_th
    omega_r_theory = np.sqrt(1.0 + 3.0 * k_ld**2)
    gamma_L_theory = -np.sqrt(np.pi / 8.0) / (k_ld**3) * np.exp(-0.5 / (k_ld**2) - 1.5)
    v_phase_theory = omega_r_theory / k
    print(f"Theoretical Langmuir frequency: omega_r = {omega_r_theory:.3f} omega_pe")
    print(f"Theoretical Phase velocity: v_phi = {v_phase_theory:.3f} v_th")
    print(f"Theoretical Landau damping rate: gamma_L = {gamma_L_theory:.4f} omega_pe")

    weight = (n0 * boxsize) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    np.random.seed(42)

    # Invert cumulative distribution to perturb spatial density: n(x) = n0 * (1 + alpha * cos(k*x))
    u = np.linspace(0.0, 1.0, N_particles, endpoint=False)
    x = u * boxsize
    for _ in range(5):
        f = (x + (alpha / k) * np.sin(k * x)) / boxsize - u
        df = (1.0 + alpha * np.cos(k * x)) / boxsize
        x -= f / df
    x = np.mod(x, boxsize)

    # Maxwellian velocity distribution
    vel = np.zeros((N_particles, 3))
    vel[:, 0] = np.random.normal(0.0, v_th, N_particles)

    solver.add_species("thermal_electrons", q=q_macro, m=m_macro, pos=x, vel=vel)
    solver.initialize()

    # Capture waveforms E(x) at t = 0, 5, 15, 25
    waveforms = {}
    record_times = [0.0, 5.0, 15.0, 25.0]

    def record_wave(s):
        for rt in record_times:
            if abs(s.time - rt) < s.dt / 2 and rt not in waveforms:
                waveforms[rt] = s.E.copy()

    record_wave(solver)

    print("\nEvolving collisionless plasma...")
    solver.run(t_end=t_end, callback=record_wave, callback_interval=1)

    time_arr = np.array(solver.history["time"])
    e_field = np.array(solver.history["E_field"])
    max_e = np.array(solver.history["max_E"])

    print("\nGenerating Creative Landau Damping Infographic...")
    os.makedirs("docs/images", exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor=EPIC_COLORS["bg_dark"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    # -------------------------------------------------------------
    # Panel 1: Peak Electric Field Oscillations & Theoretical Envelope
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    ax1.semilogy(time_arr, max_e, color="#38bdf8", lw=2.0, label="Measured $|E|_{max}(t)$")

    # Theoretical envelope
    E0 = max_e[0]
    theory_envelope = E0 * np.exp(gamma_L_theory * time_arr)
    ax1.semilogy(
        time_arr,
        theory_envelope,
        "--",
        color="#f43f5e",
        lw=2.2,
        label=rf"Landau Linear Theory: $\gamma_L = {gamma_L_theory:.3f} \, \omega_{{pe}}$",
    )
    ax1.set_xlim(0, t_end)
    ax1.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Peak Electric Field $|E|_{max}$", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) Collisionless Electric Field Damping & Linear Envelope", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 2: Kinetic Phase Space & Fine Filamentation
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    sp_x = solver.species[0].pos[:, 0]
    sp_vx = solver.species[0].vel[:, 0]
    sub = slice(None, None, 4)
    ax2.scatter(sp_x[sub], sp_vx[sub], s=0.5, color="#a855f7", alpha=0.35, label="Phase Mixed Electrons")
    # Annotate resonant phase velocity line
    ax2.axhline(v_phase_theory, color="#fbbf24", linestyle="--", lw=1.8, label=rf"Resonant $v_\phi = \omega/k \approx {v_phase_theory:.2f} \, v_{{th}}$")
    ax2.axhline(-v_phase_theory, color="#fbbf24", linestyle=":", lw=1.5)

    ax2.set_xlim(0, boxsize)
    ax2.set_ylim(-4.2, 4.2)
    ax2.set_xlabel(r"Spatial Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
    ax2.set_title(rf"(b) Kinetic Phase-Space Distribution ($t = {t_end}\ \omega_{{pe}}^{{-1}}$)", fontsize=13, fontweight="bold", color="#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax2.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 3: Spatial Waveform Damping Snapshots E(x, t)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    wf_colors = ["#38bdf8", "#34d399", "#fbbf24", "#f43f5e"]
    for rt, col in zip(record_times, wf_colors):
        if rt in waveforms:
            ax3.plot(solver.grid_x, waveforms[rt], color=col, lw=2.0, label=rf"$t = {rt:.1f} \, \omega_{{pe}}^{{-1}}$")

    ax3.set_xlim(0, boxsize)
    ax3.set_ylim(-0.24, 0.24)
    ax3.set_xlabel(r"Spatial Coordinate $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Electric Field $E(x)$", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) Wave Packet Damping Profile in Real Space", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 4: Electrostatic Wave Energy Exponential Decay
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    ax4.semilogy(time_arr, np.maximum(e_field, 1e-12), color="#34d399", lw=2.2, label=r"Measured Field Energy $\mathcal{E}_{field}(t)$")
    e_energy_theory = e_field[0] * np.exp(2.0 * gamma_L_theory * time_arr)
    ax4.semilogy(time_arr, e_energy_theory, "--", color="#fbbf24", lw=2.0, label=r"Theory: $2\gamma_L$ Rate")

    ax4.set_xlim(0, t_end)
    ax4.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Wave Energy $\mathcal{E}_{field}$", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) Field Energy Decay & Collisionless Phase Mixing", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Collisionless Landau Damping Comprehensive Benchmark",
        subtitle="Exact Bohm-Gross Frequency Oscillation, Analytic Damping Rate Envelope, and Phase-Space Mixing",
    )
    save_epic_plot(fig, "docs/images/landau_damping.png")
    print("==========================================================")


if __name__ == "__main__":
    run_landau_damping()
