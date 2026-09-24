"""
End-to-End Unit Test for 3D Particle-in-Cell Engine
===================================================
Tests 3D plasma oscillations and energy conservation.
"""

import numpy as np
from epic.solvers.pic3d import PIC3DSolver


def test_pic3d_energy_conservation():
    """Verify energy conservation in a 3D plasma oscillation benchmark."""
    Nx = 16
    Ny = 16
    Nz = 16
    Lx = 2.0 * np.pi
    Ly = 2.0 * np.pi
    Lz = 2.0 * np.pi
    n0 = 1.0
    N_part = 30000
    dt = 0.02
    t_end = 2.0

    weight = (n0 * Lx * Ly * Lz) / N_part
    q_macro = -1.0 * weight
    m_macro = 1.0 * weight

    solver = PIC3DSolver(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dt=dt)

    np.random.seed(42)
    x = np.random.uniform(0, Lx, N_part)
    y = np.random.uniform(0, Ly, N_part)
    z = np.random.uniform(0, Lz, N_part)

    # 3D sinusoidal perturbation
    x += 0.05 * np.sin(x)
    y += 0.05 * np.sin(y)
    z += 0.05 * np.sin(z)
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)
    z = np.mod(z, Lz)

    vel = np.zeros((N_part, 3))

    solver.add_species("electrons_3d", q=q_macro, m=m_macro, pos_x=x, pos_y=y, pos_z=z, vel=vel)
    solver.initialize()

    initial_energy = solver.history["E_total"][0]
    solver.run(t_end=t_end)
    final_energy = solver.history["E_total"][-1]

    rel_drift = abs(final_energy - initial_energy) / initial_energy
    assert rel_drift < 0.05, f"3D PIC energy drift exceeded threshold: {rel_drift:.2e}"


if __name__ == "__main__":
    test_pic3d_energy_conservation()
    print("3D PIC end-to-end test PASSED successfully!")
