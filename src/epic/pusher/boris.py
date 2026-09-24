"""
Vectorized Boris Particle Pusher (1D-3V, 2D-3V, 3D-3V)
======================================================
Implements the symplectic Boris algorithm for advancing charged particles
under the Lorentz force:
    d(m*v)/dt = q * (E + v x B)

References:
    Boris, J. P. (1970). Relativistic plasma simulation-optimization of a hybrid code.
    Proc. Fourth Conf. Num. Sim. Plasmas, Naval Res. Lab, Wash. D.C., 3-67.
    Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation. CRC press.
"""

from typing import Union
import numpy as np


def boris_push(
    v_in: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
    q: Union[float, np.ndarray],
    m: Union[float, np.ndarray],
    dt: float,
) -> np.ndarray:
    """Advance particle velocities from v^{n-1/2} to v^{n+1/2} using the Boris scheme.

    Parameters
    ----------
    v_in : np.ndarray
        Particle velocities at time n-1/2, shape (N, 3) or (3,).
    E : np.ndarray
        Electric field at particle positions at time n, shape (N, 3) or (3,).
    B : np.ndarray
        Magnetic field at particle positions at time n, shape (N, 3) or (3,).
    q : float or np.ndarray
        Charge of particle(s), scalar or shape (N,) or (N, 1).
    m : float or np.ndarray
        Mass of particle(s), scalar or shape (N,) or (N, 1).
    dt : float
        Time step size.

    Returns
    -------
    v_out : np.ndarray
        Updated particle velocities at time n+1/2, same shape as v_in.
    """
    v_in = np.asarray(v_in, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    is_1d_input = (v_in.ndim == 1)
    if is_1d_input:
        v_in = v_in[np.newaxis, :]
        if E.ndim == 1:
            E = E[np.newaxis, :]
        if B.ndim == 1:
            B = B[np.newaxis, :]

    # Ensure charge-to-mass ratio has shape (N, 1) if array
    qm = q / m
    if isinstance(qm, np.ndarray) and qm.ndim == 1:
        qm = qm[:, np.newaxis]

    # Half-step electric field acceleration: v^- = v^{n-1/2} + (q*E/m) * (dt/2)
    v_minus = v_in + (qm * E) * (0.5 * dt)

    # Magnetic rotation vector: t = (q*B/m) * (dt/2)
    t = (qm * B) * (0.5 * dt)

    # Magnitude squared |t|^2 for scalar denominator
    t_sq = np.sum(t * t, axis=-1, keepdims=True)

    # Rotation vector: s = 2*t / (1 + |t|^2)
    s = (2.0 * t) / (1.0 + t_sq)

    # Rotate velocity:
    # v' = v^- + (v^- x t)
    # v^+ = v^- + (v' x s)
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Second half-step electric field acceleration: v^{n+1/2} = v^+ + (q*E/m) * (dt/2)
    v_out = v_plus + (qm * E) * (0.5 * dt)

    if is_1d_input:
        return v_out[0]
    return v_out


def retard_velocity(
    v_0: np.ndarray,
    E_0: np.ndarray,
    B_0: np.ndarray,
    q: Union[float, np.ndarray],
    m: Union[float, np.ndarray],
    dt: float,
) -> np.ndarray:
    """Retard particle velocity by -dt/2 to initialize leapfrog time centering.

    Given physical initial velocities v(t=0) and fields at t=0, produces
    v(t = -dt/2) so that the standard leapfrog cycle (drift pos by dt, push vel by dt)
    remains strictly second-order accurate.
    """
    return boris_push(v_0, E_0, B_0, q, m, dt=-dt)
