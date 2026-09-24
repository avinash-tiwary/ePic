"""
Production-Grade 1D-3V Particle-in-Cell (PIC) Simulation Engine
===============================================================
A fully vectorized, symplectic 1D spatial + 3D velocity PIC solver
with spectral Poisson solving and Cloud-In-Cell (CIC) interpolation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from ..field.cic import deposit_charge_1d, interpolate_field_1d
from ..field.poisson import solve_poisson_1d_fft
from ..pusher.boris import boris_push, retard_velocity
from ..diagnostics.energy import compute_kinetic_energy, compute_field_energy_1d


class Species:
    """Container for a plasma species (e.g., electrons, ions, beams)."""

    def __init__(
        self,
        name: str,
        q: float,
        m: float,
        pos: np.ndarray,
        vel: np.ndarray,
    ):
        self.name = name
        self.q = float(q)
        self.m = float(m)
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 1)
        self.vel = np.asarray(vel, dtype=np.float64)
        if self.vel.ndim == 1:
            self.vel = self.vel[:, np.newaxis]
        if self.vel.shape[1] == 1:
            # Expand 1V to 3V
            self.vel = np.hstack((self.vel, np.zeros((len(self.vel), 2))))
        self.vel_prev = self.vel.copy()

    @property
    def N(self) -> int:
        return len(self.pos)


class PIC1DSolver:
    """1D-3V Electrostatic/Magnetostatic Particle-In-Cell Solver.

    Features:
    - Time-centered, symplectic Leapfrog-Boris particle integration.
    - Cloud-In-Cell (CIC) charge deposition and field weighting.
    - Spectral (FFT) Poisson solver with exact modified wave numbers.
    - Multi-species support (electrons, ions, drifting beams).
    - Synchronized time-centered energy conservation diagnostics.
    """

    def __init__(
        self,
        Nx: int = 400,
        boxsize: float = 45.0,
        dt: float = 0.1,
        epsilon_0: float = 1.0,
        B_ext: Optional[Union[List[float], np.ndarray]] = None,
        E_ext: Optional[Union[List[float], np.ndarray]] = None,
    ):
        self.Nx = int(Nx)
        self.boxsize = float(boxsize)
        self.dx = self.boxsize / self.Nx
        self.dt = float(dt)
        self.epsilon_0 = float(epsilon_0)

        self.B_ext = np.zeros(3) if B_ext is None else np.asarray(B_ext, dtype=np.float64)
        self.E_ext = np.zeros(3) if E_ext is None else np.asarray(E_ext, dtype=np.float64)

        self.grid_x = np.linspace(0.0, self.boxsize, self.Nx, endpoint=False)
        self.species: List[Species] = []

        # State fields
        self.time = 0.0
        self.step_count = 0
        self.rho = np.zeros(self.Nx)
        self.phi = np.zeros(self.Nx)
        self.E = np.zeros(self.Nx)

        # Diagnostics history
        self.history: Dict[str, list] = {
            "time": [],
            "E_kin": [],
            "E_field": [],
            "E_total": [],
            "max_E": [],
        }

    def add_species(
        self,
        name: str,
        q: float,
        m: float,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> Species:
        """Add a charged particle species to the simulation."""
        sp = Species(name, q, m, pos, vel)
        self.species.append(sp)
        return sp

    def _deposit_total_charge(self) -> np.ndarray:
        """Accumulate charge density from all species."""
        rho_total = np.zeros(self.Nx)
        for sp in self.species:
            rho_total += deposit_charge_1d(sp.pos, sp.q, self.Nx, self.boxsize)
        return rho_total

    def _solve_fields(self):
        """Solve Poisson equation for potential and electric field."""
        self.rho = self._deposit_total_charge()
        self.phi, self.E = solve_poisson_1d_fft(
            self.rho, self.boxsize, self.epsilon_0, use_discrete_k=True
        )

    def initialize(self):
        """Prepare simulation state and retard velocities to t = -dt/2."""
        self._solve_fields()

        # Record diagnostics at t=0 with physical initial velocities
        self._record_diagnostics(is_initial=True)

        # Retard initial velocities to v^{-1/2} to achieve second-order time-centered leapfrog
        for sp in self.species:
            E_part = interpolate_field_1d(sp.pos, self.E, self.Nx, self.boxsize)
            E_vec = np.zeros((sp.N, 3))
            E_vec[:, 0] = E_part[:, 0] + self.E_ext[0]
            E_vec[:, 1] = self.E_ext[1]
            E_vec[:, 2] = self.E_ext[2]

            sp.vel = retard_velocity(
                sp.vel, E_vec, self.B_ext, sp.q, sp.m, self.dt
            )
            sp.vel_prev = sp.vel.copy()

    def _record_diagnostics(self, is_initial: bool = False):
        """Compute and store synchronized energy and field statistics."""
        e_field = compute_field_energy_1d(self.E, self.boxsize, self.epsilon_0)

        e_kin = 0.0
        for sp in self.species:
            if is_initial:
                v_sync = sp.vel
            else:
                # Time-centered velocity at integer time n: v^n = 0.5*(v^{n-1/2} + v^{n+1/2})
                v_sync = 0.5 * (sp.vel_prev + sp.vel)
            e_kin += compute_kinetic_energy(v_sync, sp.m)

        e_total = e_kin + e_field

        self.history["time"].append(self.time)
        self.history["E_kin"].append(e_kin)
        self.history["E_field"].append(e_field)
        self.history["E_total"].append(e_total)
        self.history["max_E"].append(float(np.max(np.abs(self.E))))

    def step(self):
        """Advance the simulation by one time step dt (Drift-Scatter-Push)."""
        # 1. Drift positions by dt using velocities v^{n-1/2}
        for sp in self.species:
            sp.pos += sp.vel[:, 0:1] * self.dt
            sp.pos = np.mod(sp.pos, self.boxsize)

        self.time += self.dt
        self.step_count += 1

        # 2. Scatter: deposit charge and solve fields at t^{n}
        self._solve_fields()

        # 3. Push: advance velocities from v^{n-1/2} to v^{n+1/2}
        for sp in self.species:
            sp.vel_prev = sp.vel.copy()

            E_part = interpolate_field_1d(sp.pos, self.E, self.Nx, self.boxsize)
            E_vec = np.zeros((sp.N, 3))
            E_vec[:, 0] = E_part[:, 0] + self.E_ext[0]
            E_vec[:, 1] = self.E_ext[1]
            E_vec[:, 2] = self.E_ext[2]

            sp.vel = boris_push(
                sp.vel, E_vec, self.B_ext, sp.q, sp.m, self.dt
            )

        self._record_diagnostics(is_initial=False)

    def run(
        self,
        t_end: float,
        callback: Optional[Callable[["PIC1DSolver"], None]] = None,
        callback_interval: int = 10,
    ):
        """Run the simulation until t_end."""
        if self.step_count == 0 and len(self.history["time"]) == 0:
            self.initialize()

        while self.time < t_end - 1e-12:
            self.step()
            if callback is not None and (self.step_count % callback_interval == 0):
                callback(self)
