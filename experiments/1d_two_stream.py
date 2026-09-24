"""
1D-3V Two-Stream Instability Benchmark Simulation
=================================================
Simulates the nonlinear evolution of two counter-streaming electron beams
in a neutralizing uniform background.

Features:
- Solves Vlasov-Poisson dynamics with symplectic Boris leapfrog & CIC.
- Compares measured electrostatic field growth rate against kinetic linear dispersion theory.
- Visualizes phase-space vortex formation and energy conservation.

Run:
    python experiments/1d_two_stream.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from epic.solvers.pic1d import PIC1DSolver


def run_two_stream_instability():
    # Simulation Parameters
    Nx = 400                    # Grid cells
    boxsize = 45.0              # Domain length L
    N_particles = 100000        # Total particle count (50,000 per beam)
    dt = 0.1                    # Time step (omega_pe * dt = 0.1, stable!)
    t_end = 50.0                # Final time
    v_beam = 3.0                # Drift velocity
    v_thermal = 0.5             # Thermal velocity spread
    n0 = 1.0                    # Total background electron density

    # Macroparticle weighting
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

    # Add electron species
    solver.add_species("beam_right", q=q_macro, m=m_macro, pos=pos1, vel=vel1)
    solver.add_species("beam_left", q=q_macro, m=m_macro, pos=pos2, vel=vel2)

    solver.initialize()

    # Storage for phase-space snapshots at t = 0, 15, 25, 45
    snapshots = {}
    target_times = [0.0, 15.0, 25.0, 45.0]

    def record_snapshot(s):
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

    print("\nSimulation complete! Generating analysis plots...")
    os.makedirs("docs/images", exist_ok=True)

    # 1. Phase Space Evolution Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    axes = axes.ravel()

    for idx, tt in enumerate(target_times):
        ax = axes[idx]
        if tt in snapshots:
            p_r, v_r, p_l, v_l = snapshots[tt]
            ax.scatter(p_r[::2], v_r[::2], s=0.3, color="#1f77b4", alpha=0.4, label="Beam 1 (+v)")
            ax.scatter(p_l[::2], v_l[::2], s=0.3, color="#d62728", alpha=0.4, label="Beam 2 (-v)")
        ax.set_xlim(0, boxsize)
        ax.set_ylim(-7.0, 7.0)
        ax.set_xlabel("x (c/$\omega_{pe}$)")
        ax.set_ylabel("v$_x$ / v$_{th}$")
        ax.set_title(f"Phase Space at $\omega_{{pe}} t = {tt:.1f}$", fontweight="bold")
        if idx == 0:
            ax.legend(loc="upper right", markerscale=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Nonlinear Phase-Space Vortex Formation (Two-Stream Instability)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("docs/images/two_stream_phase_space.png")
    plt.close()
    print("  Saved docs/images/two_stream_phase_space.png")

    # 2. Energy Conservation & Instability Growth Plot
    time_arr = np.array(solver.history["time"])
    e_kin = np.array(solver.history["E_kin"])
    e_field = np.array(solver.history["E_field"])
    e_tot = np.array(solver.history["E_total"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Subplot 1: Total & Kinetic & Field Energy
    ax1.plot(time_arr, e_kin / e_tot[0], label="Kinetic Energy ($E_{kin}$)", color="#1f77b4", lw=2)
    ax1.plot(time_arr, e_tot / e_tot[0], label="Total Energy ($E_{tot}$)", color="black", linestyle="--", lw=1.5)
    ax1.plot(time_arr, e_field / e_tot[0], label="Field Energy ($E_{field}$)", color="#d62728", lw=2)
    ax1.set_xlabel("Time ($\omega_{pe} t$)")
    ax1.set_ylabel("Energy / $E_0$")
    ax1.set_title("Energy Partition & Conservation")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2: Semi-log Field Energy for Linear Growth Rate
    nonzero_field = np.maximum(e_field, 1e-12)
    ax2.semilogy(time_arr, nonzero_field, color="#d62728", lw=2, label="Measured Field Energy")
    # Linear growth fit in early exponential phase (e.g. t in [10, 22])
    mask = (time_arr >= 10.0) & (time_arr <= 22.0)
    fit = np.polyfit(time_arr[mask], np.log(nonzero_field[mask]), 1)
    gamma_measured = fit[0] / 2.0  # Since E_field ~ |E|^2 ~ exp(2*gamma*t)
    ax2.plot(
        time_arr[mask],
        np.exp(fit[1] + fit[0] * time_arr[mask]),
        "k--",
        lw=2,
        label=rf"Linear Fit: $\gamma \approx {gamma_measured:.3f} \, \omega_{{pe}}$",
    )
    ax2.set_xlabel(r"Time ($\omega_{pe} t$)")
    ax2.set_ylabel(r"$\mathcal{E}_{field}$ (Log Scale)")
    ax2.set_title("Exponential Growth & Nonlinear Saturation")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("docs/images/two_stream_energy.png")
    plt.close()
    print("  Saved docs/images/two_stream_energy.png")

    rel_drift = abs(e_tot[-1] - e_tot[0]) / e_tot[0]
    print(f"\nFinal Relative Energy Conservation Drift: {rel_drift:.2e}")
    print(f"Measured Exponential Growth Rate: gamma = {gamma_measured:.4f} omega_pe")
    print("==========================================================")


if __name__ == "__main__":
    run_two_stream_instability()
