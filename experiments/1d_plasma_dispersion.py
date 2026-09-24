"""
1D-3V Plasma Wave Dispersion Relation (k, omega) Spectral Diagnostic
=====================================================================
Directly reconstructs the kinetic plasma dispersion relation from first principles
by taking a 2D Spatio-Temporal Fourier Transform of the electric field E(x, t).

Physics:
- Langmuir Waves & Bohm-Gross dispersion: omega(k) = sqrt(omega_pe^2 + 3 * k^2 * v_th^2)
- Reconstructs dispersion surface directly from thermal kinetic fluctuations
- Hovmoller space-time diagram showing phase velocity and wave packet propagation
- Frequency spectrum slices verifying sharp resonance at plasma frequency cutoff

Run:
    python experiments/1d_plasma_dispersion.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.diagnostics import apply_epic_style, save_epic_plot, format_epic_figure, EPIC_COLORS


def run_plasma_dispersion():
    apply_epic_style()
    Nx = 256
    boxsize = 20.0 * np.pi          # L = 62.83 c/omega_pe
    v_th = 0.5                      # Thermal speed
    N_particles = 120000
    dt = 0.1
    Nt = 400                        # t_end = 40.0 omega_pe^-1

    print("==================================================================")
    print("     ePic 1D-3V PLASMA WAVE DISPERSION RELATION (k, omega)       ")
    print("==================================================================")
    print(f"Domain: L = {boxsize:.2f}, Nx = {Nx}, dx = {boxsize/Nx:.3f}")
    print(f"Particles: N = {N_particles}, v_th = {v_th}, dt = {dt}, Nt = {Nt}")

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)
    weight = (1.0 * boxsize) / N_particles

    np.random.seed(42)
    # Uniform spatial distribution with multi-mode broadband seeding
    x = np.random.uniform(0.0, boxsize, N_particles)
    # Add small coherent wave perturbations to seed specific k-modes
    k_fund = 2.0 * np.pi / boxsize
    for mode in range(1, 8):
        k_m = mode * k_fund
        x += 0.04 * (boxsize / mode) * np.sin(k_m * x + np.random.uniform(0, 2*np.pi)) / N_particles
    x = np.mod(x, boxsize)

    vel = np.random.normal(0.0, v_th, (N_particles, 3))

    solver.add_species("electrons", q=-weight, m=weight, pos=x, vel=vel)
    solver.initialize()

    # Pre-allocate 2D space-time matrix E(x, t)
    E_xt = np.zeros((Nt, Nx))

    print("\nEvolving kinetic plasma and recording space-time wavefield E(x, t)...")
    for t_step in range(Nt):
        E_xt[t_step, :] = solver.E.copy()
        solver.step()

    print("Wavefield recorded! Performing 2D Spatio-Temporal Fourier Transform...")

    # Apply 2D Hann window to reduce spectral leakage
    window_t = np.hanning(Nt)[:, np.newaxis]
    window_x = np.hanning(Nx)[np.newaxis, :]
    window_2d = window_t * window_x
    E_windowed = (E_xt - np.mean(E_xt)) * window_2d

    # 2D FFT: axis 0 is time (omega), axis 1 is space (k)
    fft_2d = np.fft.fft2(E_windowed)
    fft_shift = np.fft.fftshift(fft_2d)
    power_spectrum = np.abs(fft_shift) ** 2

    # Frequency and wavenumber axes
    omega_axis = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt)) * 2.0 * np.pi
    k_axis = np.fft.fftshift(np.fft.fftfreq(Nx, d=solver.dx)) * 2.0 * np.pi

    # Filter to positive quadrant: k > 0, omega > 0
    k_mask = (k_axis >= 0.0) & (k_axis <= 2.0)
    omega_mask = (omega_axis >= 0.0) & (omega_axis <= 3.5)

    sub_k = k_axis[k_mask]
    sub_omega = omega_axis[omega_mask]
    sub_spec = power_spectrum[np.ix_(omega_mask, k_mask)]

    # Theoretical Bohm-Gross Langmuir dispersion curve:
    # omega^2 = omega_pe^2 + 3 * k^2 * v_th^2 (omega_pe = 1.0)
    k_theory = np.linspace(0.0, 2.0, 200)
    omega_bohm_gross = np.sqrt(1.0 + 3.0 * (k_theory ** 2) * (v_th ** 2))
    # Cold plasma Langmuir cutoff
    omega_cold = np.ones_like(k_theory) * 1.0

    print("Generating Creative Multi-Panel Plasma Dispersion Dashboard...")
    os.makedirs("docs/images", exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=140, facecolor="#090d16")
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: The (k, omega) Kinetic Dispersion Diagram
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#090d16")
    log_spec = np.log10(np.maximum(sub_spec, 1e-12))
    # Normalize for stunning dynamic range
    vmax = np.percentile(log_spec, 99.8)
    vmin = vmax - 4.5

    im1 = ax1.pcolormesh(
        sub_k,
        sub_omega,
        log_spec,
        cmap="turbo",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay theoretical Bohm-Gross curve with luminous cyan glow
    ax1.plot(k_theory, omega_bohm_gross, color="#00ffff", lw=2.8, label=r"Theory: $\omega^2 = \omega_{pe}^2 + 3 k^2 v_{th}^2$")
    ax1.plot(k_theory, omega_cold, ":", color="#fbbf24", lw=2.0, label=r"Cold Cutoff: $\omega = \omega_{pe}$")

    ax1.set_xlim(0.0, 1.8)
    ax1.set_ylim(0.0, 3.2)
    ax1.set_xlabel(r"Wavenumber $k$ ($\omega_{pe} / c$)", fontsize=11, color="#f8fafc")
    ax1.set_ylabel(r"Frequency $\omega$ ($\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax1.set_title(r"(a) 2D Kinetic Spectral Power $\log_{10} |\tilde{E}(k, \omega)|^2$", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.legend(loc="upper left", framealpha=0.6, facecolor="#1e293b", edgecolor="none")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Spectral Energy Density $\log_{10} |\tilde{E}|^2$", color="#f8fafc")

    # -------------------------------------------------------------
    # Panel 2: Hovmoller Space-Time Diagram E(x, t)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#090d16")
    time_grid = np.linspace(0.0, Nt * dt, Nt)
    space_grid = np.linspace(0.0, boxsize, Nx)

    v_lim_e = np.percentile(np.abs(E_xt), 98)
    im2 = ax2.imshow(
        E_xt,
        origin="lower",
        extent=[0, boxsize, 0, Nt * dt],
        cmap="coolwarm",
        aspect="auto",
        vmin=-v_lim_e,
        vmax=v_lim_e,
    )
    ax2.set_xlabel(r"Spatial Coordinate $x$ ($c/\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax2.set_ylabel(r"Time $t$ ($\omega_{pe}^{-1}$)", fontsize=11, color="#f8fafc")
    ax2.set_title(r"(b) Hovmöller Wavefield Diagram $E(x, t)$", fontsize=13, fontweight="bold", color="#f8fafc")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Electric Field $E_x(x, t)$", color="#f8fafc")

    # -------------------------------------------------------------
    # Panel 3: Spectral Power Slices at Specific k-modes
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#090d16")
    sample_k_indices = [np.argmin(np.abs(sub_k - 0.2)), np.argmin(np.abs(sub_k - 0.5)), np.argmin(np.abs(sub_k - 0.9))]
    colors = ["#38bdf8", "#f43f5e", "#a855f7"]

    for idx, col in zip(sample_k_indices, colors):
        actual_k = sub_k[idx]
        slice_power = sub_spec[:, idx]
        slice_power /= np.max(slice_power)
        # Expected Bohm-Gross frequency
        exp_omega = np.sqrt(1.0 + 3.0 * (actual_k ** 2) * (v_th ** 2))
        ax3.plot(sub_omega, slice_power, color=col, lw=2.0, label=rf"$k = {actual_k:.2f}$ (Theory $\omega = {exp_omega:.2f}$)")
        ax3.axvline(exp_omega, color=col, linestyle="--", alpha=0.6)

    ax3.set_xlim(0.5, 2.5)
    ax3.set_xlabel(r"Frequency $\omega$ ($\omega_{pe}$)", fontsize=11, color="#f8fafc")
    ax3.set_ylabel(r"Normalized Spectral Power", fontsize=11, color="#f8fafc")
    ax3.set_title(r"(c) Resonant Frequency Peaks vs Bohm-Gross Predictions", fontsize=13, fontweight="bold", color="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    # -------------------------------------------------------------
    # Panel 4: Dispersion Relation Group & Phase Velocity
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#090d16")
    k_vals = np.linspace(0.05, 2.0, 150)
    omega_vals = np.sqrt(1.0 + 3.0 * (k_vals ** 2) * (v_th ** 2))
    v_phase = omega_vals / k_vals
    v_group = 3.0 * k_vals * (v_th ** 2) / omega_vals

    ax4.plot(k_vals, v_phase, color="#34d399", lw=2.4, label=r"Phase Velocity $v_\phi = \omega / k$")
    ax4.plot(k_vals, v_group, color="#f59e0b", lw=2.4, label=r"Group Velocity $v_g = d\omega / dk$")
    ax4.axhline(v_th, color="#e2e8f0", linestyle=":", lw=1.5, label=r"Thermal Speed $v_{th}$")

    ax4.set_xlim(0.05, 2.0)
    ax4.set_ylim(0.0, max(v_phase[0], 5.0))
    ax4.set_xlabel(r"Wavenumber $k$ ($\omega_{pe}/c$)", fontsize=11, color="#f8fafc")
    ax4.set_ylabel(r"Velocity ($c$)", fontsize=11, color="#f8fafc")
    ax4.set_title(r"(d) Kinetic Wave Speeds ($v_\phi > c$ Superluminal, $v_g \leq v_{th}$)", fontsize=13, fontweight="bold", color="#f8fafc")
    ax4.grid(True, linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper right", framealpha=0.6, facecolor="#1e293b", edgecolor="none")

    format_epic_figure(
        fig,
        title="ePic 1D-3V First-Principles Plasma Dispersion Spectral Reconstruction",
        subtitle="Verification of Langmuir Waves, Bohm-Gross Resonance, and Spatial Wavepacket Propagation",
    )
    save_epic_plot(fig, "docs/images/plasma_dispersion_relation.png")
    print("==================================================================")


if __name__ == "__main__":
    run_plasma_dispersion()
