"""
Production-Grade 2D-3V Particle-in-Cell (PIC) Simulation Engine
===============================================================
A fully vectorized, symplectic 2D spatial + 3D velocity PIC solver
with 2D spectral Poisson solving and bilinear Cloud-In-Cell (CIC) interpolation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from ..field.cic import deposit_charge_2d, interpolate_field_2d
from ..field.poisson import solve_poisson_2d_fft
from ..pusher.boris import boris_push, retard_velocity
from ..diagnostics.energy import compute_kinetic_energy, compute_field_energy_2d


class Species2D:
    """Container for a 2D-3V plasma species."""

    def __init__(
        self,
        name: str,
        q: float,
        m: float,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        vel: np.ndarray,
    ):
        self.name = name
        self.q = float(q)
        self.m = float(m)
        self.x = np.asarray(pos_x, dtype=np.float64).ravel()
        self.y = np.asarray(pos_y, dtype=np.float64).ravel()
        self.vel = np.asarray(vel, dtype=np.float64)
        if self.vel.shape[1] == 2:
            self.vel = np.hstack((self.vel, np.zeros((len(self.vel), 1))))
        self.vel_half = self.vel.copy()

    @property
    def N(self) -> int:
        return len(self.x)


class PIC2DSolver:
    """2D-3V Electrostatic/Magnetostatic Particle-In-Cell Solver.

    Features:
    - 2D spatial positions (x, y) with 3D velocities (vx, vy, vz).
    - Bilinear Cloud-In-Cell (CIC) deposition and interpolation (zero self-force).
    - 2D Spectral (FFT) Poisson solver (O(Nx Ny log(Nx Ny))).
    - Symplectic Leapfrog-Boris time integration for arbitrary B-field geometries.
    """

    def __init__(
        self,
        Nx: int = 128,
        Ny: int = 128,
        Lx: float = 20.0,
        Ly: float = 20.0,
        dt: float = 0.05,
        epsilon_0: float = 1.0,
        B_ext: Optional[Union[List[float], np.ndarray]] = None,
    ):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dt = float(dt)
        self.epsilon_0 = float(epsilon_0)

        self.B_ext = np.zeros(3) if B_ext is None else np.asarray(B_ext, dtype=np.float64)
        self.species: List[Species2D] = []

        # State fields of shape (Ny, Nx)
        self.time = 0.0
        self.step_count = 0
        self.rho = np.zeros((self.Ny, self.Nx))
        self.phi = np.zeros((self.Ny, self.Nx))
        self.Ex = np.zeros((self.Ny, self.Nx))
        self.Ey = np.zeros((self.Ny, self.Nx))

        # Diagnostics history
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
        vel: np.ndarray,
    ) -> Species2D:
        """Add a 2D charged particle species."""
        sp = Species2D(name, q, m, pos_x, pos_y, vel)
        self.species.append(sp)
        return sp

    def _deposit_total_charge(self) -> np.ndarray:
        """Accumulate 2D charge density from all species."""
        rho_total = np.zeros((self.Ny, self.Nx))
        for sp in self.species:
            rho_total += deposit_charge_2d(sp.x, sp.y, sp.q, self.Nx, self.Ny, self.Lx, self.Ly)
        return rho_total

    def _solve_fields(self):
        """Solve 2D Poisson equation via FFT."""
        self.rho = self._deposit_total_charge()
        self.phi, self.Ex, self.Ey = solve_poisson_2d_fft(
            self.rho, self.Lx, self.Ly, self.epsilon_0, use_discrete_k=True
        )

    def initialize(self):
        """Initialize fields and retard particle velocities to t = -dt/2."""
        self._solve_fields()

        # Initial energy with physical velocities v^0
        e_kin_0 = sum(compute_kinetic_energy(sp.vel, sp.m) for sp in self.species)
        e_field_0 = compute_field_energy_2d(self.Ex, self.Ey, self.Lx, self.Ly, self.epsilon_0)
        self.history["time"].append(0.0)
        self.history["E_kin"].append(e_kin_0)
        self.history["E_field"].append(e_field_0)
        self.history["E_total"].append(e_kin_0 + e_field_0)

        for sp in self.species:
            ex_p, ey_p = interpolate_field_2d(
                sp.x, sp.y, self.Ex, self.Ey, self.Nx, self.Ny, self.Lx, self.Ly
            )
            E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
            sp.vel_half = retard_velocity(sp.vel, E_vec, self.B_ext, sp.q, sp.m, self.dt)

    def step(self):
        """Advance the 2D PIC simulation by one time step dt (Kick-Sync-Drift-Field)."""
        e_kin_total = 0.0

        # 1. Kick v^{n-1/2} -> v^{n+1/2} using current fields E^n
        for sp in self.species:
            ex_p, ey_p = interpolate_field_2d(
                sp.x, sp.y, self.Ex, self.Ey, self.Nx, self.Ny, self.Lx, self.Ly
            )
            E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
            v_next = boris_push(sp.vel_half, E_vec, self.B_ext, sp.q, sp.m, self.dt)

            # Synchronized velocity at integer time n: v^n = 0.5 * (v^{n-1/2} + v^{n+1/2})
            v_n = 0.5 * (sp.vel_half + v_next)
            sp.vel = v_n.copy()
            e_kin_total += compute_kinetic_energy(v_n, sp.m)

            # 2. Position drift: x^n -> x^{n+1}
            sp.x += v_next[:, 0] * self.dt
            sp.y += v_next[:, 1] * self.dt
            sp.x = np.mod(sp.x, self.Lx)
            sp.y = np.mod(sp.y, self.Ly)

            sp.vel_half = v_next

        # Record synchronized energies at time step n
        e_field_n = compute_field_energy_2d(self.Ex, self.Ey, self.Lx, self.Ly, self.epsilon_0)
        self.history["time"].append(self.time)
        self.history["E_kin"].append(e_kin_total)
        self.history["E_field"].append(e_field_n)
        self.history["E_total"].append(e_kin_total + e_field_n)

        self.time += self.dt
        self.step_count += 1

        # 3. Update fields at t^{n+1} for the next step
        self._solve_fields()

    def run(
        self,
        t_end: float,
        callback: Optional[Callable[["PIC2DSolver"], None]] = None,
        callback_interval: int = 10,
    ):
        """Run 2D simulation up to t_end."""
        if self.step_count == 0 and len(self.history["time"]) == 0:
            self.initialize()

        while self.time < t_end - 1e-12:
            self.step()
            if callback is not None and (self.step_count % callback_interval == 0):
                callback(self)
