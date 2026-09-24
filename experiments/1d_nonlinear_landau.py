"""
1D-3V Non-linear Landau Damping & O'Neil Trapping Oscillations
=============================================================
Simulates large-amplitude electrostatic wave damping where resonant electrons
become trapped inside the wave's potential troughs, halting collisionless damping
and producing characteristic O'Neil bounce oscillations (O'Neil, 1965).

Run:
    python experiments/1d_nonlinear_landau.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_nonlinear_landau():
    apply_epic_style()

    Nx = 256
    k = 0.5
    boxsize = 2.0 * np.pi / k
    N_particles = 160000
    dt = 0.05
    t_end = 40.0
    v_th = 1.0
    alpha = 0.45                # Large amplitude -> Nonlinear regime!

    # Theoretical O'Neil bounce frequency and phase velocity
    omega_pe = 1.0
    omega_r = np.sqrt(omega_pe**2 + 3 * (k * v_th)**2)
    v_phase = omega_r / k
    # O'Neil bounce frequency estimate omega_B ~ sqrt(e k E_0 / m)
    # E0 ~ alpha / k
    omega_B = np.sqrt(k * alpha)
    tau_B = 2.0 * np.pi / omega_B

    print("==========================================================")
    print("   ePic 1D-3V NON-LINEAR LANDAU DAMPING & O'NEIL BOUNCE  ")
    print("==========================================================")
    print(f"Domain: L = {boxsize:.3f}, k = {k}, Perturbation: alpha = {alpha}")
    print(f"Theoretical O'Neil Bounce Period: tau_B ~ {tau_B:.2f} omega_pe^-1")
    print(f"Particles: N = {N_particles}, dt = {dt}, t_end = {t_end}")

    weight = (1.0 * boxsize) / N_particles
    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    np.random.seed(42)
    u = np.linspace(0.0, 1.0, N_particles, endpoint=False)
    x = u * boxsize
    for _ in range(6):
        f = (x + (alpha / k) * np.sin(k * x)) / boxsize - u
        df = (1.0 + alpha * np.cos(k * x)) / boxsize
        x -= f / df
    x = np.mod(x, boxsize)

    vel = np.random.normal(0.0, v_th, (N_particles, 3))

    solver.add_species("electrons", q=-weight, m=weight, pos=x, vel=vel)
    solver.initialize()

    print("\nEvolving non-linear kinetic trapping dynamics...")
    solver.run(t_end=t_end)

    time_arr = np.array(solver.history["time"])
    max_e = np.array(solver.history["max_E"])
    e_field = np.array(solver.history["E_field"])
    e_kin = np.array(solver.history["E_kin"])

    fig = plt.figure(figsize=(18, 11), dpi=140)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: O'Neil Bounce Oscillations & Nonlinear Saturation
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(time_arr, max_e, color=EPIC_COLORS["crimson"], lw=2.2, label=r"Measured $|E|_{max}(t)$")

    # Annotate bounce oscillation peaks
    ax1.axvline(tau_B, color=EPIC_COLORS["gold"], linestyle="--", lw=1.8, label=rf"O'Neil Period $\tau_B \approx {tau_B:.1f} \, \omega_{{pe}}^{{-1}}$")
    ax1.axvline(2 * tau_B, color=EPIC_COLORS["gold"], linestyle=":", lw=1.5)

    ax1.set_xlim(0, t_end)
    ax1.set_xlabel(r"Time ($\omega_{pe} t$)")
    ax1.set_ylabel(r"Peak Electric Field $|E|_{max}$")
    ax1.set_title(r"(a) O'Neil Bounce Oscillations & Nonlinear Saturation")
    ax1.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel 2: Phase-Space Swirl & Trapping Island
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sp_x = solver.species[0].pos[:, 0]
    sp_vx = solver.species[0].vel[:, 0]
    sub = slice(None, None, 4)
    ax2.scatter(sp_x[sub], sp_vx[sub], s=0.6, color=EPIC_COLORS["cyan"], alpha=0.35, label="Kinetic Electrons")
    ax2.axhline(v_phase, color=EPIC_COLORS["gold"], linestyle="--", lw=1.8, label=rf"Resonant $v_\phi \approx {v_phase:.2f} \, v_{{th}}$")
    ax2.axhline(-v_phase, color=EPIC_COLORS["gold"], linestyle=":", lw=1.5)

    ax2.set_xlim(0, boxsize)
    ax2.set_ylim(-4.2, 4.2)
    ax2.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"Velocity $v_x / v_{th}$")
    ax2.set_title(rf"(b) Trapped Electron Swirl in Phase Space ($t = {t_end}\ \omega_{{pe}}^{{-1}}$)")
    ax2.legend(loc="upper right", markerscale=6)

    # -------------------------------------------------------------
    # Panel 3: Self-Consistent Trapping Potential Well
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    p1, = ax3.plot(solver.grid_x, solver.phi, color=EPIC_COLORS["emerald"], lw=2.2, label=r"Potential Well $\phi(x)$")
    p2, = ax3_twin.plot(solver.grid_x, solver.E, color=EPIC_COLORS["gold"], lw=2.0, linestyle="-.", label=r"Electric Field $E(x)$")

    ax3.set_xlim(0, boxsize)
    ax3.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax3.set_ylabel(r"Electrostatic Potential $\phi$", color=EPIC_COLORS["emerald"])
    ax3_twin.set_ylabel(r"Electric Field $E$", color=EPIC_COLORS["gold"])
    ax3.set_title(r"(c) Nonlinear Potential Trough Confining Trapped Particles")
    ax3.legend(handles=[p1, p2], loc="upper right")

    # -------------------------------------------------------------
    # Panel 4: Nonlinear Wave-Particle Energy Exchange
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_arr, e_field, color=EPIC_COLORS["crimson"], lw=2.2, label=r"Wave Field Energy $\mathcal{E}_{field}(t)$")
    delta_k = e_kin - e_kin[0]
    ax4.plot(time_arr, -delta_k, color=EPIC_COLORS["cyan"], lw=2.0, linestyle="--", label=r"Kinetic Energy Transfer $-\Delta \mathcal{E}_{kin}(t)$")

    ax4.set_xlim(0, t_end)
    ax4.set_xlabel(r"Time ($\omega_{pe} t$)")
    ax4.set_ylabel(r"Energy Exchange $\Delta \mathcal{E}$")
    ax4.set_title(r"(d) Reversible Energy Sloshing between Wave and Resonant Particles")
    ax4.legend(loc="center right")

    format_epic_figure(
        fig,
        title="ePic 1D-3V Nonlinear Landau Damping & O'Neil Trapping",
        subtitle="Large-Amplitude Wave-Particle Trapping, Kinetic Bounce Cycles, and Saturation",
    )

    save_epic_plot(fig, "docs/images/nonlinear_landau_trapping.png")
    print("==========================================================")


if __name__ == "__main__":
    run_nonlinear_landau()
