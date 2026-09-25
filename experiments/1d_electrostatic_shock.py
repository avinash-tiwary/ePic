"""
1D-3V Collisionless Electrostatic Shock Wave Experiment
=======================================================
Simulates the formation, steepening, and particle reflection dynamics
of an electrostatic collisionless shock wave (Forslund & Freidberg 1971; Sorasio et al. 2006).

Key Physics:
- Supersonic plasma flow at Mach number M = v_drift / c_s > 1
- Self-consistent electrostatic potential jump Delta phi and electric shock ramp E_x
- Specular ion reflection creating the characteristic phase-space "shock foot" (v_ref = 2 v_s - v_in)
- Downstream kinetic vortex trapping and collisionless dissipation into thermal entropy

Run:
    python experiments/1d_electrostatic_shock.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.field.cic import deposit_charge_1d, interpolate_field_1d
from epic.field.poisson import solve_poisson_1d_fft
from epic.pusher.boris import boris_push, retard_velocity
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_electrostatic_shock():
    apply_epic_style()
    Nx = 400
    boxsize = 60.0                  # c / omega_pe
    dx = boxsize / Nx
    dt = 0.05
    t_end = 35.0
    Nt = int(t_end / dt)

    N_electrons = 160000
    N_ions = 160000
    m_i_ratio = 16.0                # Reduced ion-to-electron mass ratio for computational kinetic clarity

    T_e = 1.0                       # Electron temperature
    T_i = 0.05                      # Cold ion temperature (T_e / T_i >> 1 enables laminar shock)
    v_th_e = np.sqrt(T_e)
    v_th_i = np.sqrt(T_i / m_i_ratio)
    c_s = np.sqrt(T_e / m_i_ratio)  # Ion sound speed

    # Supersonic Mach number M = 2.0
    v_drift = 2.0 * c_s

    print("==================================================================")
    print("      ePic 1D-3V COLLISIONLESS ELECTROSTATIC SHOCK WAVE           ")
    print("==================================================================")
    print(f"Domain: L = {boxsize:.1f}, dx = {dx:.3f}, dt = {dt}, t_end = {t_end}")
    print(f"Mass ratio m_i/m_e = {m_i_ratio:.1f}, Ion sound speed c_s = {c_s:.3f}")
    print(f"Flow Mach number M = {v_drift / c_s:.2f} (Supersonic v_drift = {v_drift:.3f})")
    print(f"Particles: {N_electrons} electrons, {N_ions} ions")

    weight_e = (1.0 * boxsize) / N_electrons
    q_e = -1.0 * weight_e
    m_e = 1.0 * weight_e

    weight_i = (1.0 * boxsize) / N_ions
    q_i = 1.0 * weight_i
    m_i = m_i_ratio * weight_i

    np.random.seed(42)

    # Counter-streaming injection: Left half drifts right (+v_drift), Right half drifts left (-v_drift)
    # This forms a symmetric collision at x = L/2
    x_e = np.random.uniform(0.0, boxsize, N_electrons)
    x_i = np.random.uniform(0.0, boxsize, N_ions)

    v_e = np.random.normal(0.0, v_th_e, (N_electrons, 3))
    v_i = np.random.normal(0.0, v_th_i, (N_ions, 3))

    # Apply streaming drift
    left_e = x_e < (boxsize / 2.0)
    v_e[left_e, 0] += v_drift
    v_e[~left_e, 0] -= v_drift

    left_i = x_i < (boxsize / 2.0)
    v_i[left_i, 0] += v_drift
    v_i[~left_i, 0] -= v_drift

    # Initial charge deposition & Poisson solve
    rho = deposit_charge_1d(x_e, q_e, Nx, boxsize) + deposit_charge_1d(x_i, q_i, Nx, boxsize)
    phi, E = solve_poisson_1d_fft(rho, boxsize)

    E_e = interpolate_field_1d(x_e, E, Nx, boxsize)
    E_i = interpolate_field_1d(x_i, E, Nx, boxsize)

    E_e_vec = np.column_stack((E_e.ravel(), np.zeros((N_electrons, 2))))
    E_i_vec = np.column_stack((E_i.ravel(), np.zeros((N_ions, 2))))
    B_vec_e = np.zeros((N_electrons, 3))
    B_vec_i = np.zeros((N_ions, 3))

    # Retard velocities
    v_e_half = retard_velocity(v_e, E_e_vec, B_vec_e, q_e, m_e, dt)
    v_i_half = retard_velocity(v_i, E_i_vec, B_vec_i, q_i, m_i, dt)

    # History diagnostics
    time_hist = []
    max_e_hist = []
    potential_jump_hist = []

    print("\nSimulating collisionless shock ramp formation and particle reflection...")
    for step in range(Nt):
        t_now = step * dt

        # 1. Boris push
        E_e = interpolate_field_1d(x_e, E, Nx, boxsize)
        E_i = interpolate_field_1d(x_i, E, Nx, boxsize)
        E_e_vec[:, 0] = E_e.ravel()
        E_i_vec[:, 0] = E_i.ravel()

        v_e_next = boris_push(v_e_half, E_e_vec, B_vec_e, q_e, m_e, dt)
        v_i_next = boris_push(v_i_half, E_i_vec, B_vec_i, q_i, m_i, dt)

        # 2. Position advance with periodic boundaries
        x_e += v_e_next[:, 0] * dt
        x_i += v_i_next[:, 0] * dt
        x_e = np.mod(x_e, boxsize)
        x_i = np.mod(x_i, boxsize)

        # 3. Field update
        rho = deposit_charge_1d(x_e, q_e, Nx, boxsize) + deposit_charge_1d(x_i, q_i, Nx, boxsize)
        phi, E = solve_poisson_1d_fft(rho, boxsize)

        v_e_half = v_e_next
        v_i_half = v_i_next

        time_hist.append(t_now)
        max_e_hist.append(np.max(np.abs(E)))
        potential_jump_hist.append(np.max(phi) - np.min(phi))

        if (step + 1) % (Nt // 4) == 0:
            print(f"  Step {step+1}/{Nt} | Time: {(step+1)*dt:.1f} / {t_end:.1f} complete.")

    print("\nGenerating Creative Multi-Panel Electrostatic Shock Dashboard...")
    os.makedirs("docs/images", exist_ok=True)

    grid_x = np.linspace(0.0, boxsize, Nx)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor=EPIC_COLORS["bg_dark"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: Ion Phase Space Showing Shock Ramp & Reflected Foot
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    sub_i = slice(None, None, max(1, N_ions // 25000))
    # Color ions by incoming vs reflected direction
    ref_mask = (x_i[sub_i] < 30.0) & (v_i_half[sub_i, 0] < 0.0) | (x_i[sub_i] > 30.0) & (v_i_half[sub_i, 0] > 0.0)

    ax1.scatter(x_i[sub_i][~ref_mask], v_i_half[sub_i, 0][~ref_mask] / c_s, s=1.2, color="#38bdf8", alpha=0.5, label="Incoming Stream")
    ax1.scatter(x_i[sub_i][ref_mask], v_i_half[sub_i, 0][ref_mask] / c_s, s=2.2, color="#f43f5e", alpha=0.8, label="Reflected Shock Foot")

    # Annotate shock front
    ax1.axvline(boxsize / 2.0, color="#fbbf24", linestyle="--", lw=1.8, label="Collision Center")
    ax1.set_xlim(0, boxsize)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Ion Velocity $v_{i, x} / c_s$", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) Ion Kinetic Phase Space & Reflected Precursor Foot", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 2: Self-Consistent Potential Jump & Electric Shock Ramp
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    ax2_twin = ax2.twinx()

    p1, = ax2.plot(grid_x, phi, color="#34d399", lw=2.4, label=r"Potential $\phi(x)$")
    p2, = ax2_twin.plot(grid_x, E, color="#fbbf24", lw=2.0, linestyle="-.", label=r"Shock Electric Field $E_x(x)$")

    ax2.set_xlim(0, boxsize)
    ax2.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Electrostatic Potential $\phi$ ($T_e / e$)", fontsize=11, color="#34d399")
    ax2_twin.set_ylabel(r"Shock Ramp Electric Field $E_x$", fontsize=11, color="#fbbf24")
    ax2.set_title(r"(b) Electrostatic Shock Barrier $\Delta \phi$ & Electric Field Ramp", fontsize=13, fontweight="bold", color="#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax2.legend(handles=[p1, p2], loc="upper left", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 3: Electron Phase Space & Downstream Thermalization
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    sub_e = slice(None, None, max(1, N_electrons // 25000))
    ax3.scatter(x_e[sub_e], v_e_half[sub_e, 0] / v_th_e, s=0.8, color="#a855f7", alpha=0.4)
    ax3.set_xlim(0, boxsize)
    ax3.set_ylim(-3.5, 3.5)
    ax3.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Electron Velocity $v_{e, x} / v_{th, e}$", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) Electron Phase-Space Vortex & Downstream Heating", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # -------------------------------------------------------------
    # Panel 4: Potential Barrier & Ramp Field Evolution
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    ax4.plot(time_hist, potential_jump_hist, color="#06b6d4", lw=2.2, label=r"Potential Jump $\Delta \phi(t)$")
    ax4.plot(time_hist, max_e_hist, color="#f43f5e", lw=2.0, linestyle="--", label=r"Peak Shock Field $|E_{max}|(t)$")
    ax4.set_xlim(0, max(time_hist))
    ax4.set_ylim(-0.5, 8.5)
    ax4.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Shock Amplitude", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) Shock Ramp Growth & Steady-State Saturation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper left", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Collisionless Electrostatic Shock Benchmark",
        subtitle="Supersonic Stream Collision (M=2.0), Specular Ion Reflection, and Downstream Kinetic Heating",
    )
    save_epic_plot(fig, "docs/images/electrostatic_shock_dynamics.png")
    print("==================================================================")


if __name__ == "__main__":
    run_electrostatic_shock()
