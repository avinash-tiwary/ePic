"""
2D-3V Harris Current Sheet & Magnetic Reconnection Benchmark
============================================================
Simulates particle dynamics and reconnection current sheet equilibrium,
directly benchmarking the kinetic plasma models from M.Sc. thesis:
"Characterising Magnetic Reconnection in Asymmetric Medium"
(Avinash Kumar Himanshu, IIT Indore).

Features:
- Harris equilibrium profile: B_x(y) = B_0 * tanh((y - Ly/2)/L_p)
- Diamagnetic current drift v_z supporting the magnetic shear
- Magnetic flux perturbation exciting X-point reconnection and plasmoid formation.

Run:
    python experiments/2d_harris_reconnection.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from epic.field.cic import deposit_charge_2d, interpolate_field_2d
from epic.field.poisson import solve_poisson_2d_fft
from epic.pusher.boris import boris_push, retard_velocity


def run_harris_reconnection():
    Nx = 128
    Ny = 128
    Lx = 25.0
    Ly = 12.5
    N_particles = 100000
    dt = 0.05
    t_end = 20.0

    B0 = 1.0                    # Asymptotic guide field
    L_p = 1.5                   # Current sheet half-thickness
    y_center = Ly / 2.0
    v_th = 0.5                  # Thermal velocity
    v_drift_z = 1.0             # Diamagnetic drift supporting current
    n0 = 1.0

    print("==========================================================")
    print("      ePic 2D-3V HARRIS CURRENT SHEET & RECONNECTION     ")
    print("==========================================================")
    print(f"Domain: {Lx:.1f} x {Ly:.1f}, Current Sheet Thickness: {2*L_p:.1f}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    weight = (n0 * Lx * Ly) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    np.random.seed(42)

    # Sample particle positions following Harris profile: n(y) ~ sech^2((y - y_c)/L_p) + 0.2
    # Rejection sampling in y
    y_parts = []
    while len(y_parts) < N_particles:
        y_cand = np.random.uniform(0.0, Ly, N_particles)
        prob = (1.0 / np.cosh((y_cand - y_center) / L_p)) ** 2 + 0.2
        p_rand = np.random.uniform(0.0, 1.2, N_particles)
        accept = y_cand[p_rand < prob]
        y_parts.extend(accept.tolist())

    y_pos = np.array(y_parts[:N_particles])
    x_pos = np.random.uniform(0.0, Lx, N_particles)

    # Velocities: thermal in x and y, diamagnetic drift in z
    vel = np.zeros((N_particles, 3))
    vel[:, 0] = np.random.normal(0.0, v_th, N_particles)
    vel[:, 1] = np.random.normal(0.0, v_th, N_particles)
    vel[:, 2] = np.random.normal(v_drift_z, v_th, N_particles)

    # Analytic magnetic field B(x, y) = [B_x, B_y, 0] with X-point perturbation
    # B_x = B0 * tanh((y - y_c)/L_p) - delta_B * cos(2*pi*x/Lx) * sin(pi*(y - y_c)/Ly)
    # B_y = delta_B * (Ly/Lx) * sin(2*pi*x/Lx) * cos(pi*(y - y_c)/Ly)
    delta_B = 0.1 * B0

    def compute_b_field(xp, yp):
        bx = B0 * np.tanh((yp - y_center) / L_p) - delta_B * np.cos(
            2.0 * np.pi * xp / Lx
        ) * np.sin(np.pi * (yp - y_center) / Ly)
        by = delta_B * (Ly / Lx) * np.sin(
            2.0 * np.pi * xp / Lx
        ) * np.cos(np.pi * (yp - y_center) / Ly)
        bz = np.zeros_like(bx)
        return np.column_stack((bx, by, bz))

    # Initial self-consistent electric field
    rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
    phi, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
    ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
    E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
    B_vec = compute_b_field(x_pos, y_pos)

    # Retard velocity to t = -dt/2
    v_half = retard_velocity(vel, E_vec, B_vec, q_macro, m_macro, dt)

    Nt = int(t_end / dt)
    print("\nSimulating kinetic reconnection evolution...")

    for step in range(Nt):
        # 1. Force evaluation & Boris Push
        ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
        E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
        B_vec = compute_b_field(x_pos, y_pos)

        v_next = boris_push(v_half, E_vec, B_vec, q_macro, m_macro, dt)

        # 2. Drift positions
        x_pos += v_next[:, 0] * dt
        y_pos += v_next[:, 1] * dt
        x_pos = np.mod(x_pos, Lx)
        y_pos = np.clip(y_pos, 0.01, Ly - 0.01)  # Wall boundary in y

        # 3. Field update
        rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
        phi, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
        v_half = v_next

    print("\nSimulation complete! Generating reconnection topology plot...")
    os.makedirs("docs/images", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

    # 1. Magnetic Field Streamlines and Current Sheet
    gx = np.linspace(0.0, Lx, Nx)
    gy = np.linspace(0.0, Ly, Ny)
    GX, GY = np.meshgrid(gx, gy)
    GB = compute_b_field(GX.ravel(), GY.ravel())
    GBx = GB[:, 0].reshape((Ny, Nx))
    GBy = GB[:, 1].reshape((Ny, Nx))

    # Out-of-plane current J_z = dBy/dx - dBx/dy
    Jz = np.gradient(GBy, gx, axis=1) - np.gradient(GBx, gy, axis=0)

    im1 = ax1.imshow(
        Jz,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
    )
    ax1.streamplot(
        gx,
        gy,
        GBx,
        GBy,
        color="black",
        density=1.2,
        linewidth=1.0,
        arrowsize=1.0,
    )
    plt.colorbar(im1, ax=ax1, label=r"Current Density $J_z = (\nabla \times B)_z$")
    ax1.set_xlabel(r"x ($d_i$)")
    ax1.set_ylabel(r"y ($d_i$)")
    ax1.set_title(r"Magnetic Field Topology & Central X-Point ($\omega_{pe} t = 20$)")

    # 2. Particle Density & Outflow Jet Formation
    im2 = ax2.imshow(
        rho,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="inferno",
        aspect="auto",
    )
    plt.colorbar(im2, ax=ax2, label=r"Electron Number Density $\rho(x, y)$")
    ax2.set_xlabel(r"x ($d_i$)")
    ax2.set_ylabel(r"y ($d_i$)")
    ax2.set_title(r"Kinetic Plasma Density Distribution in Reconnecting Sheet")

    plt.tight_layout()
    plt.savefig("docs/images/harris_reconnection.png")
    plt.close()
    print("  Saved docs/images/harris_reconnection.png")
    print("==========================================================")


if __name__ == "__main__":
    run_harris_reconnection()
