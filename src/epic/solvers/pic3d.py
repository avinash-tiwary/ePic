"""
Production-Grade 3D-3V Particle-in-Cell (PIC) Simulation Engine
===============================================================
A fully vectorized, symplectic 3D spatial + 3D velocity PIC solver
with 3D spectral Poisson solving and trilinear Cloud-In-Cell (CIC) interpolation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from ..field.cic import deposit_charge_3d, interpolate_field_3d
from ..field.poisson import solve_poisson_3d_fft
from ..pusher.boris import boris_push, retard_velocity
from ..diagnostics.energy import compute_kinetic_energy, compute_field_energy_3d


class Species3D:
    """Container for a 3D-3V plasma species."""

    def __init__(
        self,
        name: str,
        q: float,
        m: float,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel: np.ndarray,
    ):
        self.name = name
        self.q = float(q)
        self.m = float(m)
        self.x = np.asarray(pos_x, dtype=np.float64).ravel()
        self.y = np.asarray(pos_y, dtype=np.float64).ravel()
        self.z = np.asarray(pos_z, dtype=np.float64).ravel()
        self.vel = np.asarray(vel, dtype=np.float64)
        self.vel_half = self.vel.copy()

    @property
    def N(self) -> int:
        return len(self.x)


class PIC3DSolver:
    """3D-3V Electrostatic/Magnetostatic Particle-In-Cell Solver.

    Features:
    - 3D spatial positions (x, y, z) with 3D velocities (vx, vy, vz).
    - Trilinear Cloud-In-Cell (CIC) deposition and interpolation (zero self-force).
    - 3D Spectral (FFT) Poisson solver (O(Nx Ny Nz log(Nx Ny Nz))).
    - Symplectic Leapfrog-Boris time integration for arbitrary B-field geometries.
    """

    def __init__(
        self,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Lx: float = 10.0,
        Ly: float = 10.0,
        Lz: float = 10.0,
        dt: float = 0.05,
        epsilon_0: float = 1.0,
        B_ext: Optional[Union[List[float], np.ndarray]] = None,
    ):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.dt = float(dt)
        self.epsilon_0 = float(epsilon_0)

        self.B_ext = np.zeros(3) if B_ext is None else np.asarray(B_ext, dtype=np.float64)
        self.species: List[Species3D] = []

        # State fields of shape (Nz, Ny, Nx)
        self.time = 0.0
        self.step_count = 0
        self.rho = np.zeros((self.Nz, self.Ny, self.Nx))
        self.phi = np.zeros((self.Nz, self.Ny, self.Nx))
        self.Ex = np.zeros((self.Nz, self.Ny, self.Nx))
        self.Ey = np.zeros((self.Nz, self.Ny, self.Nx))
        self.Ez = np.zeros((self.Nz, self.Ny, self.Nx))

        self.history: Dict[str, list] = {
            "time": [],
            "E_kin": [],
            "E_field": [],
            "E_total": [],
        }

    def add_species(
        self,
        name: str,
        q: float,
        m: float,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel: np.ndarray,
    ) -> Species3D:
        """Add a 3D charged particle species."""
        sp = Species3D(name, q, m, pos_x, pos_y, pos_z, vel)
        self.species.append(sp)
        return sp

    def _deposit_total_charge(self) -> np.ndarray:
        """Accumulate 3D charge density from all species."""
        rho_total = np.zeros((self.Nz, self.Ny, self.Nx))
        for sp in self.species:
            rho_total += deposit_charge_3d(
                sp.x, sp.y, sp.z, sp.q, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz
            )
        return rho_total

    def _solve_fields(self):
        """Solve 3D Poisson equation via FFT."""
        self.rho = self._deposit_total_charge()
        self.phi, self.Ex, self.Ey, self.Ez = solve_poisson_3d_fft(
            self.rho, self.Lx, self.Ly, self.Lz, self.epsilon_0, use_discrete_k=True
        )

    def initialize(self):
        """Initialize fields and retard particle velocities to t = -dt/2."""
        self._solve_fields()

        e_kin_0 = sum(compute_kinetic_energy(sp.vel, sp.m) for sp in self.species)
        e_field_0 = compute_field_energy_3d(
            self.Ex, self.Ey, self.Ez, self.Lx, self.Ly, self.Lz, self.epsilon_0
        )
        self.history["time"].append(0.0)
        self.history["E_kin"].append(e_kin_0)
        self.history["E_field"].append(e_field_0)
        self.history["E_total"].append(e_kin_0 + e_field_0)

        for sp in self.species:
            ex_p, ey_p, ez_p = interpolate_field_3d(
                sp.x, sp.y, sp.z, self.Ex, self.Ey, self.Ez,
                self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz
            )
            E_vec = np.column_stack((ex_p, ey_p, ez_p))
            sp.vel_half = retard_velocity(sp.vel, E_vec, self.B_ext, sp.q, sp.m, self.dt)

    def step(self):
        """Advance the 3D PIC simulation by one time step dt."""
        e_kin_total = 0.0

        for sp in self.species:
            ex_p, ey_p, ez_p = interpolate_field_3d(
                sp.x, sp.y, sp.z, self.Ex, self.Ey, self.Ez,
                self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz
            )
            E_vec = np.column_stack((ex_p, ey_p, ez_p))
            v_next = boris_push(sp.vel_half, E_vec, self.B_ext, sp.q, sp.m, self.dt)

            v_n = 0.5 * (sp.vel_half + v_next)
            e_kin_total += compute_kinetic_energy(v_n, sp.m)

            sp.x += v_next[:, 0] * self.dt
            sp.y += v_next[:, 1] * self.dt
            sp.z += v_next[:, 2] * self.dt
            sp.x = np.mod(sp.x, self.Lx)
            sp.y = np.mod(sp.y, self.Ly)
            sp.z = np.mod(sp.z, self.Lz)

            sp.vel_half = v_next

        e_field_n = compute_field_energy_3d(
            self.Ex, self.Ey, self.Ez, self.Lx, self.Ly, self.Lz, self.epsilon_0
        )
        self.history["time"].append(self.time)
        self.history["E_kin"].append(e_kin_total)
        self.history["E_field"].append(e_field_n)
        self.history["E_total"].append(e_kin_total + e_field_n)

        self.time += self.dt
        self.step_count += 1

        self._solve_fields()

    def run(
        self,
        t_end: float,
        callback: Optional[Callable[["PIC3DSolver"], None]] = None,
        callback_interval: int = 10,
    ):
        """Run 3D simulation up to t_end."""
        if self.step_count == 0 and len(self.history["time"]) == 0:
            self.initialize()

        while self.time < t_end - 1e-12:
            self.step()
            if callback is not None and (self.step_count % callback_interval == 0):
                callback(self)
