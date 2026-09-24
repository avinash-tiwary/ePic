"""
End-to-End Unit Test for 1D Particle-in-Cell Engine
===================================================
Tests cold plasma oscillation and energy conservation.
"""

import numpy as np
from epic.solvers.pic1d import PIC1DSolver


def test_pic1d_energy_conservation():
    """Verify second-order energy conservation in a plasma oscillation benchmark."""
    Nx = 128
    boxsize = 2.0 * np.pi
    n0 = 1.0
    N_part = 20000
    dt = 0.02
    t_end = 6.28  # One full plasma oscillation period

    # Macroparticle weight
    weight = n0 * boxsize / N_part
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    x = np.linspace(0, boxsize, N_part, endpoint=False)
    x += 0.05 * np.sin(x)
    x = np.mod(x, boxsize)
    v = np.zeros((N_part, 3))

    solver.add_species("electrons", q=q_macro, m=m_macro, pos=x, vel=v)
    solver.initialize()

    initial_energy = solver.history["E_total"][0]
    solver.run(t_end=t_end)
    final_energy = solver.history["E_total"][-1]

    # Verify total energy is conserved within 0.1% (rel drift < 1e-3)
    rel_drift = abs(final_energy - initial_energy) / initial_energy
    assert rel_drift < 1e-3, f"1D PIC energy drift exceeded threshold: {rel_drift:.2e}"


if __name__ == "__main__":
    test_pic1d_energy_conservation()
    print("1D PIC end-to-end test PASSED successfully!")
