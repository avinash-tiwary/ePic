# ePic: Vectorized 1D / 2D / 3D Particle-in-Cell Plasma Physics Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-6%2F6%20passing-brightgreen.svg)]()
[![Energy Conservation](https://img.shields.io/badge/Energy%20Drift-%3C%200.01%25-success.svg)]()

**ePic** is a high-performance, modular **Particle-in-Cell (PIC)** plasma simulation engine written in vectorized Python. It resolves the collisionless kinetic dynamics of charged particles governed by the coupled **Vlasov–Poisson** and **Vlasov–Maxwell** systems across 1D-3V, 2D-3V, and 3D-3V geometries.

Originally developed at the **Department of Astronomy, Astrophysics and Space Engineering, Indian Institute of Technology Indore (IIT Indore)** in conjunction with research on magnetic reconnection and plasmoid instability (*M.Sc. Thesis, Avinash Kumar Himanshu, 2023*), `ePic` has been re-architected into a production-grade scientific computing suite.

---

## Benchmark Highlights & Physical Validations

### 1. 1D-3V Two-Stream Instability & Phase-Space Eddies
When two counter-streaming electron beams ($v = \pm 3.0$) interact, electrostatic perturbations grow exponentially, leading to non-linear particle trapping and phase-space vortex roll-up.

| Nonlinear Phase-Space Vortex Roll-up | Energy Partition & Exponential Growth Rate |
| :---: | :---: |
| ![Two-Stream Phase Space](docs/images/two_stream_phase_space.png) | ![Two-Stream Energy](docs/images/two_stream_energy.png) |

- **Growth Rate Validation**: Measured numerical growth rate $\gamma \approx 0.237\,\omega_{pe}$ agrees precisely with kinetic linear dispersion theory for warm beams ($v_{th} = 0.5$).
- **Energy Conservation**: Relative energy drift $\Delta E / E_0 = 6.8 \times 10^{-5}$ ($< 0.01\%$) over 500 time steps.

---

### 2. 1D-3V Collisionless Landau Damping
A fundamental kinetic phenomenon where Langmuir waves in a thermal Maxwellian plasma damp without physical particle collisions via resonant wave-particle energy exchange.

<p align="center">
  <img src="docs/images/landau_damping.png" width="70%" alt="Landau Damping Decay Curve" />
</p>

- **Theoretical Rate**:
  $$\gamma_L = -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3} \exp\left(-\frac{1}{2(k\lambda_D)^2} - \frac{3}{2}\right) \approx -0.151\,\omega_{pe}$$
- The measured peak electric field envelope $|E|_{\max}(t)$ follows the analytic Landau decay envelope down to the thermal noise floor.

---

### 3. 2D-3V Filamentation & Beam Instability
2D electrostatic simulation showing the emergence of transverse spatial filamentation and oblique wave modes alongside longitudinal velocity vortices.

<p align="center">
  <img src="docs/images/2d_filamentation.png" width="85%" alt="2D Filamentation & Phase Space" />
</p>

---

### 4. 2D-3V Harris Current Sheet & Magnetic Reconnection
Simulates kinetic magnetic reconnection in a neutral plasma sheet with diamagnetic drift supporting the magnetic shear:
$$\mathbf{B}(y) = B_0 \tanh\left(\frac{y - y_c}{L_p}\right) \hat{\mathbf{x}} + \delta \mathbf{B}(x, y)$$

<p align="center">
  <img src="docs/images/harris_reconnection.png" width="85%" alt="Harris Current Sheet Reconnection" />
</p>

Directly benchmarks the kinetic reconnection models and plasmoid generation studied in the author's Master's thesis under Dr. Bhargav Vaidya.

---

### 5. 3D-3V Spherical Plasma Expansion
3D electrostatic simulation of a localized plasma cloud expanding into vacuum under self-consistent Coulomb repulsions.

<p align="center">
  <img src="docs/images/3d_plasma_expansion.png" width="85%" alt="3D Spherical Plasma Expansion" />
</p>

---

## Core Computational Architecture

```
                      ┌────────────────────────────────────────┐
                      │          Particle Positions x^n        │
                      └──────────────────┬─────────────────────┘
                                         │  Cloud-in-Cell (CIC)
                                         ▼  Charge Deposition
                      ┌────────────────────────────────────────┐
                      │          Charge Density rho^n          │
                      └──────────────────┬─────────────────────┘
                                         │  Spectral Poisson Solver
                                         ▼  grad^2 phi = -rho / eps_0
                      ┌────────────────────────────────────────┐
                      │    Potential phi^n & Field E^n (Grid)  │
                      └──────────────────┬─────────────────────┘
                                         │  Inverse CIC Interpolation
                                         ▼  (Adjoint, Zero Self-Force)
                      ┌────────────────────────────────────────┐
                      │          Field E(x_p) at Particles     │
                      └──────────────────┬─────────────────────┘
                                         │  Symplectic Boris Pusher
                                         ▼  v^{n-1/2} -> v^{n+1/2}, x^n -> x^{n+1}
                      ┌────────────────────────────────────────┐
                      │    Updated Positions & Synchronized    │
                      │    Energy Conservation Diagnostics     │
                      └────────────────────────────────────────┘
```

### 1. Vectorized Boris Particle Pusher
Solves the relativistic or non-relativistic Lorentz equation:
$$\frac{d\mathbf{v}}{dt} = \frac{q}{m} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)$$
- **Symplectic Leapfrog Time-Centering**: Velocity is retarded by $-\Delta t/2$ at $t=0$, guaranteeing exact second-order global convergence $\mathcal{O}(\Delta t^2)$.
- **Exact Vector Denominator**: Eliminates legacy vector bugs by computing the scalar denominator $1 + |\mathbf{t}|^2 = 1 + \mathbf{t}\cdot\mathbf{t}$, preserving the magnetic invariant $|\mathbf{v}| = \text{const}$ to machine precision ($< 10^{-12}$).

### 2. Spectral (FFT) Poisson Solver
- Replaces legacy $\mathcal{O}(N^6)$ dense matrix inversion with $\mathcal{O}(N \log N)$ Fast Fourier Transforms.
- Uses exact finite-difference discrete Laplacian eigenvalues:
  $$k_{\text{eff}}^2 = \left(\frac{2}{\Delta x}\sin\frac{k_x \Delta x}{2}\right)^2 + \left(\frac{2}{\Delta y}\sin\frac{k_y \Delta y}{2}\right)^2 + \left(\frac{2}{\Delta z}\sin\frac{k_z \Delta z}{2}\right)^2$$
- Unconditionally gauge-invariant: strictly sets the spatial average $\hat{\phi}(0) = 0$ to eliminate numerical gauge drift.

### 3. Volume-Conserving Cloud-in-Cell (CIC)
- Implemented via vectorized index histograms (`np.bincount` / `np.add.at`).
- Dual deposition and interpolation kernels guarantee **exact momentum conservation and zero numerical self-force**.

---

## Repository Structure

```
ePic/
├── pyproject.toml                     # Modern build & packaging configuration
├── requirements.txt                   # Production dependencies
├── docs/
│   ├── images/                        # High-resolution benchmark figures
│   └── thesis/                        # Master of Science thesis (IIT Indore, 2023)
├── src/
│   └── epic/                          # Core Python package
│       ├── __init__.py
│       ├── pusher/
│       │   └── boris.py               # Vectorized 1D/2D/3D Boris particle pusher
│       ├── field/
│       │   ├── cic.py                 # 1D, 2D, 3D Cloud-In-Cell routines
│       │   └── poisson.py             # 1D, 2D, 3D Spectral FFT Poisson solvers
│       ├── diagnostics/
│       │   └── energy.py              # Synchronized kinetic & field energy tracking
│       └── solvers/
│           ├── pic1d.py               # Production 1D-3V PIC engine
│           ├── pic2d.py               # Production 2D-3V PIC engine
│           └── pic3d.py               # Production 3D-3V PIC engine
├── experiments/                       # Executable benchmark scripts
│   ├── 1d_two_stream.py               # Two-stream instability + dispersion relation
│   ├── 1d_landau_damping.py           # Collisionless Landau damping
│   ├── 2d_two_stream.py               # 2D filamentation instability
│   ├── 2d_harris_reconnection.py      # Harris sheet magnetic reconnection
│   └── 3d_plasma_expansion.py         # 3D spherical plasma expansion
├── notebooks/                         # Interactive tutorials
│   ├── 01_Boris_Pusher_Verification.ipynb
│   ├── 02_1D_Two_Stream_Instability.ipynb
│   ├── 03_1D_Landau_Damping.ipynb
│   └── legacy/                        # Archived historical thesis development notebooks
└── tests/                             # Automated test suite (6/6 passing)
    ├── test_boris.py                  # Energy conservation in pure B & cyclotron orbits
    ├── test_cic.py                    # Charge conservation & zero self-force
    ├── test_poisson.py                # Analytic sinusoid verification
    ├── test_1d_pic.py                 # 1D end-to-end energy conservation
    ├── test_2d_pic.py                 # 2D end-to-end energy conservation
    └── test_3d_pic.py                 # 3D end-to-end energy conservation
```

---

## Installation & Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/avinash-tiwary/ePic.git
cd ePic

# Using standard venv + pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Or using uv (recommended for speed)
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Quickstart

Run a 1D plasma simulation in 10 lines of Python:

```python
import numpy as np
from epic.solvers.pic1d import PIC1DSolver

# 1. Initialize solver
solver = PIC1DSolver(Nx=256, boxsize=45.0, dt=0.1)

# 2. Add two counter-streaming electron beams
N = 100000
weight = 45.0 / N
pos1 = np.random.uniform(0.0, 45.0, N // 2)
vel1 = np.random.normal(3.0, 0.5, (N // 2, 3))

pos2 = np.random.uniform(0.0, 45.0, N // 2)
vel2 = np.random.normal(-3.0, 0.5, (N // 2, 3))

solver.add_species("beam_pos", q=-weight, m=weight, pos=pos1, vel=vel1)
solver.add_species("beam_neg", q=-weight, m=weight, pos=pos2, vel=vel2)

# 3. Evolve Vlasov-Poisson dynamics
solver.initialize()
solver.run(t_end=40.0)

print(f"Energy drift: {abs(solver.history['E_total'][-1] - solver.history['E_total'][0]) / solver.history['E_total'][0]:.2e}")
```

---

## Running the Benchmark Experiments

Run any of the ready-to-use physics benchmark experiments:

```bash
# 1D Two-stream instability (produces phase space vortex and growth rate)
python experiments/1d_two_stream.py

# 1D Collisionless Landau damping (matches analytic gamma_L)
python experiments/1d_landau_damping.py

# 2D Two-stream & filamentation instability
python experiments/2d_two_stream.py

# 2D Harris current sheet & magnetic reconnection (MSc thesis benchmark)
python experiments/2d_harris_reconnection.py

# 3D Spherical plasma expansion
python experiments/3d_plasma_expansion.py
```

---

## Running the Unit Tests

Execute the full automated test suite:

```bash
python tests/test_boris.py
python tests/test_cic.py
python tests/test_poisson.py
python tests/test_1d_pic.py
python tests/test_2d_pic.py
python tests/test_3d_pic.py
```

---

## Scientific References

1. **Boris, J. P.** (1970). *Relativistic plasma simulation-optimization of a hybrid code*. Proc. Fourth Conf. Num. Sim. Plasmas, Naval Res. Lab, Wash. D.C., 3-67.
2. **Birdsall, C. K., & Langdon, A. B.** (2004). *Plasma Physics via Computer Simulation*. CRC Press / Taylor & Francis.
3. **Hockney, R. W., & Eastwood, J. W.** (1988). *Computer Simulation Using Particles*. Adam Hilger, Bristol.
4. **Himanshu, A. K.** (2023). *Characterising Magnetic Reconnection in Asymmetric Medium*. M.Sc. Thesis, Department of Astronomy, Astrophysics and Space Engineering, Indian Institute of Technology Indore.
5. **Derouillat, J., et al.** (2018). *SMILEI: A collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation*. Computer Physics Communications, 222, 351-373.
