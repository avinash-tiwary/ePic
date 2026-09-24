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

from epic.solvers.pic3d import PIC3DSolver


def run_3d_plasma_expansion():
    Nx = 32
    Ny = 32
    Nz = 32
    Lx = 20.0
    Ly = 20.0
    Lz = 20.0
    N_particles = 40000
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

    print("\nSimulation complete! Generating 3D slice visualization...")
    os.makedirs("docs/images", exist_ok=True)

    # Midplane slice (z = Lz/2)
    mid_z = Nz // 2
    rho_midplane = solver.rho[mid_z, :, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    # 1. 2D Slice of 3D Charge Density
    im1 = ax1.imshow(
        rho_midplane,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    plt.colorbar(im1, ax=ax1, label=r"Charge Density $\rho(x, y, z=L_z/2)$")
    ax1.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax1.set_ylabel(r"y ($c/\omega_{pe}$)")
    ax1.set_title(r"3D Expansion Midplane Slice ($z = L_z/2$)")

    # 2. 3D Particle Scatter Preview (subsample)
    sp = solver.species[0]
    sub = slice(None, None, 20)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        sp.x[sub],
        sp.y[sub],
        sp.z[sub],
        s=1.0,
        c=np.linalg.norm(sp.vel[sub], axis=1),
        cmap="plasma",
        alpha=0.6,
    )
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Ly)
    ax2.set_zlim(0, Lz)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("3D Particle Position & Kinetic Velocity")

    plt.tight_layout()
    plt.savefig("docs/images/3d_plasma_expansion.png")
    plt.close()
    print("  Saved docs/images/3d_plasma_expansion.png")
    print("==========================================================")


if __name__ == "__main__":
    run_3d_plasma_expansion()
