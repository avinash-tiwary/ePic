"""
ePic: Vectorized 1D/2D/3D Particle-In-Cell (PIC) Plasma Physics Suite
=====================================================================

A modular, high-performance computational plasma physics toolkit featuring:
- Symplectic Boris-Pusher (1D-3V, 2D-3V, 3D-3V)
- Cloud-In-Cell (CIC) charge deposition and force weighting (1D, 2D, 3D)
- Gauge-invariant Fast Fourier Transform (FFT) Poisson solvers
- Benchmark experiments: Two-Stream Instability, Landau Damping, Plasma Oscillations,
  2D Filamentation, Harris Sheet Magnetic Reconnection, and 3D Plasma Expansion.

Author: Avinash Kumar Himanshu
Department of Astronomy, Astrophysics and Space Engineering
Indian Institute of Technology Indore
"""

__version__ = "2.0.0"
__author__ = "Avinash Kumar Himanshu"

from .pusher.boris import boris_push, retard_velocity
from .field.cic import (
    deposit_charge_1d,
    interpolate_field_1d,
    deposit_charge_2d,
    interpolate_field_2d,
    deposit_charge_3d,
    interpolate_field_3d,
)
from .field.poisson import (
    solve_poisson_1d_fft,
    solve_poisson_2d_fft,
    solve_poisson_3d_fft,
)
from .solvers.pic1d import PIC1DSolver, Species
from .solvers.pic2d import PIC2DSolver, Species2D
from .solvers.pic3d import PIC3DSolver, Species3D

__all__ = [
    "boris_push",
    "retard_velocity",
    "deposit_charge_1d",
    "interpolate_field_1d",
    "deposit_charge_2d",
    "interpolate_field_2d",
    "deposit_charge_3d",
    "interpolate_field_3d",
    "solve_poisson_1d_fft",
    "solve_poisson_2d_fft",
    "solve_poisson_3d_fft",
    "PIC1DSolver",
    "Species",
    "PIC2DSolver",
    "Species2D",
    "PIC3DSolver",
    "Species3D",
]
