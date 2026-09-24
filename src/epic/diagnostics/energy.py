"""
Energy and Conservation Diagnostics for Particle-in-Cell Simulations
====================================================================
Tracks particle kinetic energy, electrostatic field energy, and total energy
conservation to verify code fidelity and numerical stability.
"""

from typing import Union
import numpy as np


def compute_kinetic_energy(
    vel: np.ndarray,
    m: Union[float, np.ndarray],
) -> float:
    """Compute total particle kinetic energy: E_kin = 0.5 * sum(m * |v|^2)."""
    v_sq = np.sum(vel * vel, axis=-1)
    if np.isscalar(m):
        return 0.5 * float(m) * float(np.sum(v_sq))
    return 0.5 * float(np.sum(np.asarray(m).ravel() * v_sq))


def compute_field_energy_1d(
    E_grid: np.ndarray,
    boxsize: float,
    epsilon_0: float = 1.0,
) -> float:
    """Compute 1D electrostatic field energy: E_field = 0.5 * epsilon_0 * integral(E^2 dx)."""
    Nx = len(E_grid)
    dx = boxsize / Nx
    return 0.5 * epsilon_0 * dx * float(np.sum(E_grid * E_grid))


def compute_field_energy_2d(
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    Lx: float,
    Ly: float,
    epsilon_0: float = 1.0,
) -> float:
    """Compute 2D electrostatic field energy: E_field = 0.5 * epsilon_0 * integral(|E|^2 dA)."""
    Ny, Nx = Ex_grid.shape
    dx = Lx / Nx
    dy = Ly / Ny
    return 0.5 * epsilon_0 * (dx * dy) * float(np.sum(Ex_grid**2 + Ey_grid**2))


def compute_field_energy_3d(
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Lx: float,
    Ly: float,
    Lz: float,
    epsilon_0: float = 1.0,
) -> float:
    """Compute 3D electrostatic field energy: E_field = 0.5 * epsilon_0 * integral(|E|^2 dV)."""
    Nz, Ny, Nx = Ex_grid.shape
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    return 0.5 * epsilon_0 * (dx * dy * dz) * float(np.sum(Ex_grid**2 + Ey_grid**2 + Ez_grid**2))
