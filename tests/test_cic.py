"""
Unit tests for Cloud-in-Cell (CIC) Deposition and Interpolation
==============================================================
Verifies global charge conservation and zero numerical self-force.
"""

import numpy as np
from epic.field.cic import deposit_charge_1d, interpolate_field_1d, deposit_charge_2d, interpolate_field_2d
from epic.field.poisson import solve_poisson_1d_fft


def test_charge_conservation_1d():
    """Verify that sum(rho * dx) equals sum(q_i) to machine precision."""
    Nx = 100
    boxsize = 20.0
    dx = boxsize / Nx
    N_part = 10000

    x = np.random.uniform(0, boxsize, N_part)
    q = np.random.uniform(-1.0, 1.0, N_part)

    rho = deposit_charge_1d(x, q, Nx, boxsize)
    total_grid_charge = np.sum(rho) * dx
    total_part_charge = np.sum(q)

    rel_diff = abs(total_grid_charge - total_part_charge) / abs(total_part_charge)
    assert rel_diff < 1e-12, f"1D charge conservation violated: rel diff = {rel_diff}"


def test_zero_self_force_1d():
    """Verify that an isolated particle in a periodic box feels zero net self-force."""
    Nx = 64
    boxsize = 10.0
    x_part = np.array([3.456])
    q_part = 1.0

    rho = deposit_charge_1d(x_part, q_part, Nx, boxsize)
    _, E_grid = solve_poisson_1d_fft(rho, boxsize, epsilon_0=1.0)
    E_part = interpolate_field_1d(x_part, E_grid, Nx, boxsize)

    # Self-force must be zero to machine precision (typically < 1e-14)
    assert abs(E_part[0, 0]) < 1e-12, f"Non-zero self-force: E = {E_part[0, 0]}"


def test_charge_conservation_2d():
    """Verify 2D bilinear CIC conserves total charge."""
    Nx = 32
    Ny = 32
    Lx = 10.0
    Ly = 10.0
    N_part = 5000

    x = np.random.uniform(0, Lx, N_part)
    y = np.random.uniform(0, Ly, N_part)
    q = np.random.uniform(0.5, 2.0, N_part)

    rho = deposit_charge_2d(x, y, q, Nx, Ny, Lx, Ly)
    dx = Lx / Nx
    dy = Ly / Ny
    total_grid_charge = np.sum(rho) * (dx * dy)
    total_part_charge = np.sum(q)

    rel_diff = abs(total_grid_charge - total_part_charge) / total_part_charge
    assert rel_diff < 1e-12, f"2D charge conservation violated: rel diff = {rel_diff}"


if __name__ == "__main__":
    test_charge_conservation_1d()
    test_zero_self_force_1d()
    test_charge_conservation_2d()
    print("All CIC tests PASSED successfully!")
