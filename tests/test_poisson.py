"""
Unit tests for Spectral (FFT) Poisson Solvers
=============================================
Verifies exact analytic solutions for 1D and 2D Poisson equations.
"""

import numpy as np
from epic.field.poisson import solve_poisson_1d_fft, solve_poisson_2d_fft


def test_poisson_1d_analytic_sinusoid():
    """Verify 1D solver against analytic: rho(x) = sin(2*pi*m*x/L) -> phi(x) = rho / k^2."""
    Nx = 128
    L = 2.0 * np.pi
    m = 2
    x = np.linspace(0.0, L, Nx, endpoint=False)
    k = 2.0 * np.pi * m / L

    rho = np.sin(k * x)
    phi_analytic = rho / (k ** 2)

    # With use_discrete_k=False to test against continuous differential equation
    phi_num, _ = solve_poisson_1d_fft(rho, L, epsilon_0=1.0, use_discrete_k=False)

    max_err = np.max(np.abs(phi_num - phi_analytic))
    assert max_err < 1e-12, f"1D Poisson analytic mismatch: error = {max_err}"


def test_poisson_2d_analytic_sinusoid():
    """Verify 2D solver against analytic: rho(x,y) = sin(kx*x)*cos(ky*y)."""
    Nx = 64
    Ny = 64
    Lx = 2.0 * np.pi
    Ly = 2.0 * np.pi
    mx, my = 2, 3
    kx = 2.0 * np.pi * mx / Lx
    ky = 2.0 * np.pi * my / Ly

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    rho = np.sin(kx * X) * np.cos(ky * Y)
    phi_analytic = rho / (kx**2 + ky**2)

    phi_num, _, _ = solve_poisson_2d_fft(rho, Lx, Ly, epsilon_0=1.0, use_discrete_k=False)

    max_err = np.max(np.abs(phi_num - phi_analytic))
    assert max_err < 1e-12, f"2D Poisson analytic mismatch: error = {max_err}"


if __name__ == "__main__":
    test_poisson_1d_analytic_sinusoid()
    test_poisson_2d_analytic_sinusoid()
    print("All Poisson solver tests PASSED successfully!")
