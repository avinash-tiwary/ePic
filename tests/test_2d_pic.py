"""
End-to-End Unit Test for 2D Particle-in-Cell Engine
===================================================
Tests 2D plasma oscillations and energy conservation.
"""

import numpy as np
from epic.solvers.pic2d import PIC2DSolver


def test_pic2d_energy_conservation():
    """Verify energy conservation in a 2D plasma oscillation benchmark."""
    Nx = 64
    Ny = 64
    Lx = 2.0 * np.pi
    Ly = 2.0 * np.pi
    n0 = 1.0
    N_part = 25000
    dt = 0.02
    t_end = 2.0

    weight = (n0 * Lx * Ly) / N_part
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC2DSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dt=dt)

    np.random.seed(42)
    x = np.random.uniform(0, Lx, N_part)
    y = np.random.uniform(0, Ly, N_part)
    # Small 2D wave perturbation
    x += 0.05 * np.sin(x)
    y += 0.05 * np.cos(y)
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)

    vel = np.zeros((N_part, 3))

    solver.add_species("electrons_2d", q=q_macro, m=m_macro, pos_x=x, pos_y=y, vel=vel)
    solver.initialize()

    initial_energy = solver.history["E_total"][0]
    solver.run(t_end=t_end)
    final_energy = solver.history["E_total"][-1]

    rel_drift = abs(final_energy - initial_energy) / initial_energy
    assert rel_drift < 0.015, f"2D PIC energy drift exceeded threshold: {rel_drift:.2e}"


if __name__ == "__main__":
    test_pic2d_energy_conservation()
    print("2D PIC end-to-end test PASSED successfully!")
