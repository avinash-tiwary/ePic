"""
1D-3V Collisionless Landau Damping Benchmark Simulation
======================================================
Simulates the collisionless damping of an electrostatic Langmuir wave in a
thermal Maxwellian plasma, comparing numerical decay against linear Landau theory.

Theory:
    Linear Landau damping rate:
        gamma_L = - sqrt(pi/8) * (omega_pe / (k * lambda_D)^3) * exp(-1/(2*(k*lambda_D)^2) - 1.5)

Run:
    python experiments/1d_landau_damping.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from epic.solvers.pic1d import PIC1DSolver


def run_landau_damping():
    Nx = 256
    k = 0.5
    boxsize = 2.0 * np.pi / k    # L = 4*pi ~ 12.566
    N_particles = 150000        # High particle count to suppress thermal noise
    dt = 0.05
    t_end = 30.0
    v_th = 1.0                  # Thermal velocity -> lambda_D = v_th / omega_pe = 1.0
    n0 = 1.0
    alpha = 0.05                # Perturbation amplitude (linear regime)

    print("==========================================================")
    print("        ePic 1D-3V COLLISIONLESS LANDAU DAMPING          ")
    print("==========================================================")
    print(f"Domain: L = {boxsize:.4f}, k = {k}, k*lambda_D = {k*v_th:.2f}")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    # Analytic Landau damping rate
    k_ld = k * v_th
    omega_r_theory = np.sqrt(1.0 + 3.0 * k_ld**2)
    gamma_L_theory = -np.sqrt(np.pi / 8.0) / (k_ld**3) * np.exp(-0.5 / (k_ld**2) - 1.5)
    print(f"Theoretical Langmuir frequency: omega_r = {omega_r_theory:.3f} omega_pe")
    print(f"Theoretical Landau damping rate: gamma_L = {gamma_L_theory:.4f} omega_pe")

    # Macroparticle weighting
    weight = (n0 * boxsize) / N_particles
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    np.random.seed(42)

    # Invert cumulative distribution to perturb spatial density: n(x) = n0 * (1 + alpha * cos(k*x))
    # Cumulative: F(x) = (x + alpha/k * sin(k*x)) / L
    # We solve F(x) = u via Newton-Raphson
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

    print("\nEvolving collisionless plasma...")
    solver.run(t_end=t_end)

    time_arr = np.array(solver.history["time"])
    e_field = np.array(solver.history["E_field"])
    max_e = np.array(solver.history["max_E"])

    os.makedirs("docs/images", exist_ok=True)

    plt.figure(figsize=(9, 5), dpi=120)
    plt.semilogy(time_arr, max_e, color="#1f77b4", lw=1.8, label="PIC Measured $|E|_{max}(t)$")

    # Theoretical envelope: E_0 * exp(gamma_L * t)
    E0 = max_e[0]
    theory_envelope = E0 * np.exp(gamma_L_theory * time_arr)
    plt.semilogy(
        time_arr,
        theory_envelope,
        "r--",
        lw=2,
        label=rf"Linear Landau Theory: $\gamma_L = {gamma_L_theory:.3f} \, \omega_{{pe}}$",
    )

    plt.xlabel(r"Time ($\omega_{pe} t$)")
    plt.ylabel(r"Peak Electric Field $|E|_{max}$")
    plt.title("Collisionless Landau Damping of Langmuir Wave", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/images/landau_damping.png")
    plt.close()
    print("\n  Saved docs/images/landau_damping.png")
    print("==========================================================")


if __name__ == "__main__":
    run_landau_damping()
