"""
2D-3V Two-Stream & Filamentation Instability Benchmark Simulation
================================================================
Simulates 2D electrostatic beam-beam interaction showing 2D phase-space vortex
formation, transverse filamentation, and 2D spatial Fourier modal analysis.

Run:
    python experiments/2d_two_stream.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic2d import PIC2DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_2d_two_stream():
    apply_epic_style()
    Nx = 128
    Ny = 128
    Lx = 30.0
    Ly = 30.0
    N_particles = 120000
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

    print("\nGenerating Creative 2D Filamentation Master Dashboard...")
    os.makedirs("docs/images", exist_ok=True)

    # Compute 2D Spatial Fourier Power Spectrum of density rho(x, y)
    rho_2d = solver.rho
    rho_fft = np.fft.fftshift(np.fft.fft2(rho_2d - np.mean(rho_2d)))
    power_2d = np.abs(rho_fft) ** 2
    log_power = np.log10(np.maximum(power_2d, 1e-10))

    kx_axis = np.fft.fftshift(np.fft.fftfreq(Nx, d=solver.dx)) * 2.0 * np.pi
    ky_axis = np.fft.fftshift(np.fft.fftfreq(Ny, d=solver.dy)) * 2.0 * np.pi

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor="#090d16")
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: 2D Plasma Density Filaments rho(x, y)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    im1 = ax1.imshow(
        solver.rho,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="inferno",
        aspect="auto",
    )
    ax1.set_xlabel(r"Streamwise Coordinate $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Transverse Coordinate $y$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) 2D Transverse Charge Filaments $\rho(x, y)$", fontsize=13, fontweight="bold", color="#f8fafc")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Charge Density $\rho(x, y)$", color="#f8fafc")

    # -------------------------------------------------------------
    # Panel 2: 2D Spatial Fourier Power Spectrum |rho(kx, ky)|^2
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    im2 = ax2.imshow(
        log_power,
        origin="lower",
        extent=[kx_axis[0], kx_axis[-1], ky_axis[0], ky_axis[-1]],
        cmap="turbo",
        aspect="auto",
    )
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_xlabel(r"Streamwise Wavenumber $k_x$ ($\omega_{pe}/c$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Transverse Wavenumber $k_y$ ($\omega_{pe}/c$)", fontsize=11, color="#f8fafc")
    ax2.set_title(r"(b) 2D Spatial Fourier Spectrum $\log_{10} |\tilde{\rho}(k_x, k_y)|^2$", fontsize=13, fontweight="bold", color="#f8fafc")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Spectral Power $\log_{10} |\tilde{\rho}|^2$", color="#f8fafc")

    # -------------------------------------------------------------
    # Panel 3: Phase Space Projection (x vs vx)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    p_x1 = solver.species[0].x
    p_vx1 = solver.species[0].vel[:, 0]
    p_x2 = solver.species[1].x
    p_vx2 = solver.species[1].vel[:, 0]
    sub = slice(None, None, 3)

    ax3.scatter(p_x1[sub], p_vx1[sub], s=0.4, color="#38bdf8", alpha=0.35, label="Beam 1 (+v)")
    ax3.scatter(p_x2[sub], p_vx2[sub], s=0.4, color="#f43f5e", alpha=0.35, label="Beam 2 (-v)")
    ax3.set_xlim(0, Lx)
    ax3.set_ylim(-6.5, 6.5)
    ax3.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Velocity $v_x / v_{th}$", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) 2D Kinetic Phase-Space Projection $(x, v_x)$", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="upper right", markerscale=8, framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 4: Energy History & Exponential Growth
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    t_hist = np.array(solver.history["time"])
    e_field = np.array(solver.history["E_field"])
    e_kin = np.array(solver.history["E_kin"])
    e_tot = np.array(solver.history["E_total"])

    ax4.semilogy(t_hist, np.maximum(e_field, 1e-12), color="#34d399", lw=2.2, label=r"Electrostatic Field $\mathcal{E}_{field}(t)$")
    ax4.plot(t_hist, e_kin, color="#38bdf8", lw=1.8, label=r"Kinetic Energy $\mathcal{E}_{kin}(t)$")
    ax4.plot(t_hist, e_tot, color="#f8fafc", linestyle="--", lw=1.5, label=r"Total Energy $\mathcal{E}_{tot}(t)$")

    ax4.set_xlim(0, t_end)
    ax4.set_xlabel(r"Time ($\omega_{pe} t$)", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Energy (Arbitrary Units)", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) 2D Energy Partition & Growth Saturation", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, which="both", linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="lower right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 2D-3V Two-Stream & Transverse Filamentation Benchmark",
        subtitle="Spatial Current Channels, 2D Fourier Wavevector Spectrum, and Phase-Space Vortex Coalescence",
    )

    save_epic_plot(fig, "docs/images/2d_filamentation.png")
    print("==========================================================")


if __name__ == "__main__":
    run_2d_two_stream()
