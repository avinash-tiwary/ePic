"""
2D-3V Two-Stream & Filamentation Instability Benchmark Simulation
================================================================
Simulates 2D electrostatic beam-beam interaction showing 2D phase-space vortex
formation and transverse filamentation.

Run:
    python experiments/2d_two_stream.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from epic.solvers.pic2d import PIC2DSolver


def run_2d_two_stream():
    Nx = 128
    Ny = 128
    Lx = 30.0
    Ly = 30.0
    N_particles = 100000
    dt = 0.08
    t_end = 25.0
    v_beam = 2.5
    v_th = 0.4
    n0 = 1.0

    print("==========================================================")
    print("      ePic 2D-3V TWO-STREAM & FILAMENTATION BENCHMARK    ")
    print("==========================================================")
    print(f"Grid: {Nx}x{Ny}, Domain: {Lx}x{Ly}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    weight = (n0 * Lx * Ly) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC2DSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dt=dt)

    Nh = N_particles // 2
    np.random.seed(42)

    # Beam 1 (+vx)
    x1 = np.random.uniform(0.0, Lx, Nh)
    y1 = np.random.uniform(0.0, Ly, Nh)
    vel1 = np.zeros((Nh, 3))
    vel1[:, 0] = np.random.normal(v_beam, v_th, Nh)
    vel1[:, 1] = np.random.normal(0.0, v_th, Nh)

    # Beam 2 (-vx)
    x2 = np.random.uniform(0.0, Lx, Nh)
    y2 = np.random.uniform(0.0, Ly, Nh)
    vel2 = np.zeros((Nh, 3))
    vel2[:, 0] = np.random.normal(-v_beam, v_th, Nh)
    vel2[:, 1] = np.random.normal(0.0, v_th, Nh)

    solver.add_species("beam_pos", q=q_macro, m=m_macro, pos_x=x1, pos_y=y1, vel=vel1)
    solver.add_species("beam_neg", q=q_macro, m=m_macro, pos_x=x2, pos_y=y2, vel=vel2)

    solver.initialize()

    print("\nEvolving 2D Vlasov-Poisson system...")
    solver.run(t_end=t_end)

    print("\nGenerating 2D filamentation and phase-space visualizations...")
    os.makedirs("docs/images", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

    # 1. 2D Charge Density Heatmap with Streamlines
    im = ax1.imshow(
        solver.rho,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="inferno",
        aspect="auto",
    )
    plt.colorbar(im, ax=ax1, label=r"Charge Density $\rho(x, y)$")
    ax1.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax1.set_ylabel(r"y ($c/\omega_{pe}$)")
    ax1.set_title(r"2D Plasma Charge Distribution $\rho(x, y)$ at $\omega_{pe} t = 25$")

    # 2. Phase Space Slice (x vs vx)
    p_x1 = solver.species[0].x
    p_vx1 = solver.species[0].vel[:, 0]
    p_x2 = solver.species[1].x
    p_vx2 = solver.species[1].vel[:, 0]

    ax2.scatter(p_x1[::3], p_vx1[::3], s=0.4, color="#1f77b4", alpha=0.3, label="Beam 1 (+v)")
    ax2.scatter(p_x2[::3], p_vx2[::3], s=0.4, color="#d62728", alpha=0.3, label="Beam 2 (-v)")
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(-6.0, 6.0)
    ax2.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"v$_x$ / v$_{th}$")
    ax2.set_title(r"2D Phase Space Projection $(x, v_x)$ at $\omega_{pe} t = 25$")
    ax2.legend(loc="upper right", markerscale=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("docs/images/2d_filamentation.png")
    plt.close()
    print("  Saved docs/images/2d_filamentation.png")

    e_tot = np.array(solver.history["E_total"])
    drift = abs(e_tot[-1] - e_tot[0]) / e_tot[0]
    print(f"2D Relative Energy Conservation Drift: {drift:.2e}")
    print("==========================================================")


if __name__ == "__main__":
    run_2d_two_stream()
