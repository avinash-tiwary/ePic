"""
Unit tests for the Vectorized Boris Particle Pusher
===================================================
Verifies exact energy conservation and cyclotron orbits in magnetic fields.
"""

import numpy as np
from epic.pusher.boris import boris_push, retard_velocity


def test_boris_energy_conservation_pure_b():
    """Verify that pure magnetic deflection conserves kinetic energy to machine precision."""
    q = 1.0
    m = 1.0
    dt = 0.05
    N_steps = 1000

    v0 = np.array([1.5, 0.8, -0.5])
    B = np.array([0.2, 0.5, 1.2])  # General 3D magnetic field
    E = np.array([0.0, 0.0, 0.0])

    initial_energy = 0.5 * m * np.sum(v0**2)
    v = v0.copy()

    for _ in range(N_steps):
        v = boris_push(v, E, B, q, m, dt)

    final_energy = 0.5 * m * np.sum(v**2)
    rel_error = abs(final_energy - initial_energy) / initial_energy
    assert rel_error < 1e-12, f"Kinetic energy drift in pure B: {rel_error:.2e}"


def test_boris_cyclotron_frequency():
    """Verify that the cyclotron orbit matches analytic period T_c = 2*pi / omega_c."""
    q = 1.0
    m = 1.0
    Bz = 2.0
    B = np.array([0.0, 0.0, Bz])
    E = np.array([0.0, 0.0, 0.0])

    omega_c = q * Bz / m
    T_c = 2.0 * np.pi / omega_c

    N_steps = 1000
    dt = T_c / N_steps

    v0 = np.array([1.0, 0.0, 0.0])
    v = v0.copy()

    # Evolve for exactly one period
    for _ in range(N_steps):
        v = boris_push(v, E, B, q, m, dt)

    # After one full period, velocity should return to initial
    vel_error = np.linalg.norm(v - v0)
    assert vel_error < 1e-2, f"Cyclotron velocity mismatch: {vel_error}"


def test_vectorized_multiple_particles():
    """Verify that vectorized push matches single-particle pushes across heterogeneous fields."""
    N = 500
    q = np.random.uniform(0.5, 2.0, N)
    m = np.random.uniform(0.8, 1.5, N)
    dt = 0.01

    v = np.random.randn(N, 3)
    E = np.random.randn(N, 3)
    B = np.random.randn(N, 3)

    v_vec = boris_push(v, E, B, q, m, dt)

    for i in range(10):  # Check first 10 individually
        v_single = boris_push(v[i], E[i], B[i], q[i], m[i], dt)
        assert np.allclose(v_vec[i], v_single, atol=1e-12)


if __name__ == "__main__":
    test_boris_energy_conservation_pure_b()
    test_boris_cyclotron_frequency()
    test_vectorized_multiple_particles()
    print("All Boris pusher tests PASSED successfully!")
