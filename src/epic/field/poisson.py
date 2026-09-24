"""
Gauge-Invariant Spectral (FFT) Poisson Solvers for 1D and 2D Periodic Geometries
================================================================================
Solves:
    nabla^2 phi = - (rho - rho_mean) / epsilon_0
    E = - nabla phi

Using discrete Fourier transforms (FFT) with exact modified wave numbers.
Guarantees:
- Zero self-force on particles
- Exact periodic boundary conditions
- Unconditional numerical stability and O(N log N) computational complexity.
"""

from typing import Tuple
import numpy as np


def solve_poisson_1d_fft(
    rho: np.ndarray,
    boxsize: float,
    epsilon_0: float = 1.0,
    use_discrete_k: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 1D Poisson equation on a uniform periodic grid using FFT.

    Parameters
    ----------
    rho : np.ndarray
        Charge density at grid points, shape (Nx,).
    boxsize : float
        Domain length L.
    epsilon_0 : float, default 1.0
        Permittivity of free space.
    use_discrete_k : bool, default True
        If True, uses the exact finite-difference discrete Laplacian eigenvalues
        k_eff^2 = (2/dx * sin(k*dx/2))^2 to ensure exact consistency with
        finite-difference gradient operators and eliminate self-force.

    Returns
    -------
    phi : np.ndarray
        Electrostatic potential on grid, shape (Nx,).
    E : np.ndarray
        Electric field on grid, shape (Nx,).
    """
    Nx = len(rho)
    dx = boxsize / Nx

    # Neutralize charge: rho' = rho - <rho>
    rho_neutral = rho - np.mean(rho)

    # Real FFT
    rho_hat = np.fft.rfft(rho_neutral)
    m = np.arange(len(rho_hat))
    k = (2.0 * np.pi * m) / boxsize

    if use_discrete_k:
        # Exact discrete Laplacian eigenvalue
        k_sq = (2.0 / dx * np.sin(0.5 * k * dx)) ** 2
    else:
        k_sq = k ** 2

    # Invert Poisson equation in Fourier space
    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    # k=0 mode is 0 (fixing the electrostatic potential gauge to mean(phi) = 0)
    phi_hat[1:] = (rho_hat[1:] / epsilon_0) / k_sq[1:]

    # Electric field in Fourier space: E_hat = -i * k_diff * phi_hat
    if use_discrete_k:
        # Central-difference derivative eigenvalue: i * sin(k*dx) / dx
        k_diff = np.sin(k * dx) / dx
    else:
        k_diff = k

    E_hat = -1j * k_diff * phi_hat

    # Transform back to real space
    phi = np.fft.irfft(phi_hat, n=Nx)
    E = np.fft.irfft(E_hat, n=Nx)

    return phi, E


def solve_poisson_2d_fft(
    rho: np.ndarray,
    Lx: float,
    Ly: float,
    epsilon_0: float = 1.0,
    use_discrete_k: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D Poisson equation on a uniform periodic grid using 2D real FFT.

    Parameters
    ----------
    rho : np.ndarray
        2D charge density array of shape (Ny, Nx).
    Lx, Ly : float
        Domain sizes in x and y directions.
    epsilon_0 : float, default 1.0
        Permittivity of free space.
    use_discrete_k : bool, default True
        Whether to use discrete Laplacian eigenvalues.

    Returns
    -------
    phi : np.ndarray
        2D potential array of shape (Ny, Nx).
    Ex : np.ndarray
        2D electric field in x of shape (Ny, Nx).
    Ey : np.ndarray
        2D electric field in y of shape (Ny, Nx).
    """
    Ny, Nx = rho.shape
    dx = Lx / Nx
    dy = Ly / Ny

    rho_neutral = rho - np.mean(rho)
    rho_hat = np.fft.rfft2(rho_neutral)

    kx = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)

    if use_discrete_k:
        k_sq = (
            (2.0 / dx * np.sin(0.5 * KX * dx)) ** 2
            + (2.0 / dy * np.sin(0.5 * KY * dy)) ** 2
        )
        diff_kx = np.sin(KX * dx) / dx
        diff_ky = np.sin(KY * dy) / dy
    else:
        k_sq = KX ** 2 + KY ** 2
        diff_kx = KX
        diff_ky = KY

    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    nonzero = k_sq > 0
    phi_hat[nonzero] = (rho_hat[nonzero] / epsilon_0) / k_sq[nonzero]

    Ex_hat = -1j * diff_kx * phi_hat
    Ey_hat = -1j * diff_ky * phi_hat

    phi = np.fft.irfft2(phi_hat, s=(Ny, Nx))
    Ex = np.fft.irfft2(Ex_hat, s=(Ny, Nx))
    Ey = np.fft.irfft2(Ey_hat, s=(Ny, Nx))

    return phi, Ex, Ey


def solve_poisson_3d_fft(
    rho: np.ndarray,
    Lx: float,
    Ly: float,
    Lz: float,
    epsilon_0: float = 1.0,
    use_discrete_k: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve 3D Poisson equation on a uniform periodic grid using 3D real FFT.

    Parameters
    ----------
    rho : np.ndarray
        3D charge density array of shape (Nz, Ny, Nx).
    Lx, Ly, Lz : float
        Domain sizes in x, y, and z directions.

    Returns
    -------
    phi, Ex, Ey, Ez : np.ndarray
        3D potential and electric field arrays of shape (Nz, Ny, Nx).
    """
    Nz, Ny, Nx = rho.shape
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    rho_neutral = rho - np.mean(rho)
    rho_hat = np.fft.rfftn(rho_neutral)

    kx = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")

    if use_discrete_k:
        k_sq = (
            (2.0 / dx * np.sin(0.5 * KX * dx)) ** 2
            + (2.0 / dy * np.sin(0.5 * KY * dy)) ** 2
            + (2.0 / dz * np.sin(0.5 * KZ * dz)) ** 2
        )
        diff_kx = np.sin(KX * dx) / dx
        diff_ky = np.sin(KY * dy) / dy
        diff_kz = np.sin(KZ * dz) / dz
    else:
        k_sq = KX**2 + KY**2 + KZ**2
        diff_kx = KX
        diff_ky = KY
        diff_kz = KZ

    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    nonzero = k_sq > 0
    phi_hat[nonzero] = (rho_hat[nonzero] / epsilon_0) / k_sq[nonzero]

    Ex_hat = -1j * diff_kx * phi_hat
    Ey_hat = -1j * diff_ky * phi_hat
    Ez_hat = -1j * diff_kz * phi_hat

    phi = np.fft.irfftn(phi_hat, s=(Nz, Ny, Nx), axes=(0, 1, 2))
    Ex = np.fft.irfftn(Ex_hat, s=(Nz, Ny, Nx), axes=(0, 1, 2))
    Ey = np.fft.irfftn(Ey_hat, s=(Nz, Ny, Nx), axes=(0, 1, 2))
    Ez = np.fft.irfftn(Ez_hat, s=(Nz, Ny, Nx), axes=(0, 1, 2))

    return phi, Ex, Ey, Ez
