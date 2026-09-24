"""
Cloud-in-Cell (CIC) Charge Deposition and Field Interpolation
============================================================
First-order b-spline shape functions for volume-conserving, zero-self-force
particle-mesh mapping in periodic geometries.

References:
    Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation.
    Hockney, R. W., & Eastwood, J. W. (1988). Computer simulation using particles.
"""

from typing import Tuple, Union
import numpy as np


def deposit_charge_1d(
    x: np.ndarray,
    q: Union[float, np.ndarray],
    Nx: int,
    boxsize: float,
) -> np.ndarray:
    """Deposit particle charges onto a 1D uniform periodic grid via CIC.

    Parameters
    ----------
    x : np.ndarray
        Particle positions, shape (N,) or (N, 1).
    q : float or np.ndarray
        Particle charges. If float, all particles have charge q. If array, shape (N,).
    Nx : int
        Number of grid cells.
    boxsize : float
        Domain length L.

    Returns
    -------
    rho : np.ndarray
        Charge density at grid points, shape (Nx,).
    """
    x = np.asarray(x).ravel()
    dx = boxsize / Nx

    # Normalized grid coordinates
    xi = x / dx
    i = np.floor(xi).astype(np.int64)
    fx = xi - i

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)

    w0 = 1.0 - fx
    w1 = fx

    if np.isscalar(q):
        weights0 = q * w0
        weights1 = q * w1
    else:
        q_arr = np.asarray(q).ravel()
        weights0 = q_arr * w0
        weights1 = q_arr * w1

    # High-speed histogram accumulation
    rho = np.bincount(i0, weights=weights0, minlength=Nx)
    rho += np.bincount(i1, weights=weights1, minlength=Nx)

    # Convert charge per cell to charge density rho = Q / dx
    rho /= dx
    return rho


def interpolate_field_1d(
    x: np.ndarray,
    E_grid: np.ndarray,
    Nx: int,
    boxsize: float,
) -> np.ndarray:
    """Interpolate electric field from grid points to particle positions (Inverse CIC).

    Parameters
    ----------
    x : np.ndarray
        Particle positions, shape (N,) or (N, 1).
    E_grid : np.ndarray
        Electric field on grid vertices, shape (Nx,).
    Nx : int
        Number of grid cells.
    boxsize : float
        Domain length L.

    Returns
    -------
    E_part : np.ndarray
        Electric field at particle positions, shape (N, 1).
    """
    x = np.asarray(x).ravel()
    dx = boxsize / Nx

    xi = x / dx
    i = np.floor(xi).astype(np.int64)
    fx = xi - i

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)

    E_part = (1.0 - fx) * E_grid[i0] + fx * E_grid[i1]
    return E_part[:, np.newaxis]


def deposit_charge_2d(
    x: np.ndarray,
    y: np.ndarray,
    q: Union[float, np.ndarray],
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """Deposit particle charges onto a 2D uniform periodic grid via bilinear CIC.

    Parameters
    ----------
    x, y : np.ndarray
        Particle coordinates, shapes (N,).
    q : float or np.ndarray
        Particle charges.
    Nx, Ny : int
        Number of cells in x and y.
    Lx, Ly : float
        Physical domain extents.

    Returns
    -------
    rho : np.ndarray
        Charge density array of shape (Ny, Nx).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    dx = Lx / Nx
    dy = Ly / Ny

    xi = x / dx
    yj = y / dy

    i = np.floor(xi).astype(np.int64)
    j = np.floor(yj).astype(np.int64)

    fx = xi - i
    fy = yj - j

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)
    j0 = np.mod(j, Ny)
    j1 = np.mod(j + 1, Ny)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    if np.isscalar(q):
        q00 = q * w00
        q10 = q * w10
        q01 = q * w01
        q11 = q * w11
    else:
        q_arr = np.asarray(q).ravel()
        q00 = q_arr * w00
        q10 = q_arr * w10
        q01 = q_arr * w01
        q11 = q_arr * w11

    # Map 2D (j, i) to 1D index: k = j * Nx + i
    k00 = j0 * Nx + i0
    k10 = j0 * Nx + i1
    k01 = j1 * Nx + i0
    k11 = j1 * Nx + i1

    total_cells = Nx * Ny
    rho_flat = np.bincount(k00, weights=q00, minlength=total_cells)
    rho_flat += np.bincount(k10, weights=q10, minlength=total_cells)
    rho_flat += np.bincount(k01, weights=q01, minlength=total_cells)
    rho_flat += np.bincount(k11, weights=q11, minlength=total_cells)

    rho = rho_flat.reshape((Ny, Nx)) / (dx * dy)
    return rho


def interpolate_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate 2D vector field from grid to particle positions (Bilinear CIC).

    Parameters
    ----------
    x, y : np.ndarray
        Particle positions, shapes (N,).
    Ex_grid, Ey_grid : np.ndarray
        Electric field arrays of shape (Ny, Nx).

    Returns
    -------
    Ex_part, Ey_part : np.ndarray
        Field components at particles, shapes (N,).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    dx = Lx / Nx
    dy = Ly / Ny

    xi = x / dx
    yj = y / dy

    i = np.floor(xi).astype(np.int64)
    j = np.floor(yj).astype(np.int64)

    fx = xi - i
    fy = yj - j

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)
    j0 = np.mod(j, Ny)
    j1 = np.mod(j + 1, Ny)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    Ex_part = (
        w00 * Ex_grid[j0, i0]
        + w10 * Ex_grid[j0, i1]
        + w01 * Ex_grid[j1, i0]
        + w11 * Ex_grid[j1, i1]
    )
    Ey_part = (
        w00 * Ey_grid[j0, i0]
        + w10 * Ey_grid[j0, i1]
        + w01 * Ey_grid[j1, i0]
        + w11 * Ey_grid[j1, i1]
    )

    return Ex_part, Ey_part


def deposit_charge_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    q: Union[float, np.ndarray],
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> np.ndarray:
    """Deposit particle charges onto a 3D uniform periodic grid via trilinear CIC.

    Returns
    -------
    rho : np.ndarray
        Charge density array of shape (Nz, Ny, Nx).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    xi = x / dx
    yj = y / dy
    zk = z / dz

    i = np.floor(xi).astype(np.int64)
    j = np.floor(yj).astype(np.int64)
    k = np.floor(zk).astype(np.int64)

    fx = xi - i
    fy = yj - j
    fz = zk - k

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)
    j0 = np.mod(j, Ny)
    j1 = np.mod(j + 1, Ny)
    k0 = np.mod(k, Nz)
    k1 = np.mod(k + 1, Nz)

    wx0, wx1 = 1.0 - fx, fx
    wy0, wy1 = 1.0 - fy, fy
    wz0, wz1 = 1.0 - fz, fz

    weights = [
        (wx0 * wy0 * wz0, i0, j0, k0),
        (wx1 * wy0 * wz0, i1, j0, k0),
        (wx0 * wy1 * wz0, i0, j1, k0),
        (wx1 * wy1 * wz0, i1, j1, k0),
        (wx0 * wy0 * wz1, i0, j0, k1),
        (wx1 * wy0 * wz1, i1, j0, k1),
        (wx0 * wy1 * wz1, i0, j1, k1),
        (wx1 * wy1 * wz1, i1, j1, k1),
    ]

    total_cells = Nx * Ny * Nz
    rho_flat = np.zeros(total_cells, dtype=np.float64)

    q_arr = np.asarray(q).ravel() if not np.isscalar(q) else float(q)

    for w, ii, jj, kk in weights:
        idx = (kk * Ny + jj) * Nx + ii
        part_weight = q_arr * w if not np.isscalar(q) else q * w
        rho_flat += np.bincount(idx, weights=part_weight, minlength=total_cells)

    rho = rho_flat.reshape((Nz, Ny, Nx)) / (dx * dy * dz)
    return rho


def interpolate_field_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    Ex_grid: np.ndarray,
    Ey_grid: np.ndarray,
    Ez_grid: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate 3D vector field from grid to particle positions (Trilinear CIC)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    xi = x / dx
    yj = y / dy
    zk = z / dz

    i = np.floor(xi).astype(np.int64)
    j = np.floor(yj).astype(np.int64)
    k = np.floor(zk).astype(np.int64)

    fx = xi - i
    fy = yj - j
    fz = zk - k

    i0 = np.mod(i, Nx)
    i1 = np.mod(i + 1, Nx)
    j0 = np.mod(j, Ny)
    j1 = np.mod(j + 1, Ny)
    k0 = np.mod(k, Nz)
    k1 = np.mod(k + 1, Nz)

    wx0, wx1 = 1.0 - fx, fx
    wy0, wy1 = 1.0 - fy, fy
    wz0, wz1 = 1.0 - fz, fz

    Ex_p = np.zeros_like(x)
    Ey_p = np.zeros_like(x)
    Ez_p = np.zeros_like(x)

    weights = [
        (wx0 * wy0 * wz0, i0, j0, k0),
        (wx1 * wy0 * wz0, i1, j0, k0),
        (wx0 * wy1 * wz0, i0, j1, k0),
        (wx1 * wy1 * wz0, i1, j1, k0),
        (wx0 * wy0 * wz1, i0, j0, k1),
        (wx1 * wy0 * wz1, i1, j0, k1),
        (wx0 * wy1 * wz1, i0, j1, k1),
        (wx1 * wy1 * wz1, i1, j1, k1),
    ]

    for w, ii, jj, kk in weights:
        Ex_p += w * Ex_grid[kk, jj, ii]
        Ey_p += w * Ey_grid[kk, jj, ii]
        Ez_p += w * Ez_grid[kk, jj, ii]

    return Ex_p, Ey_p, Ez_p
