# ePic: Vectorized 1D / 2D / 3D Particle-in-Cell Plasma Physics Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)]()
[![Energy Conservation](https://img.shields.io/badge/Energy%20Drift-%3C%200.01%25-success.svg)]()
[![Style: Deep Space Dark](https://img.shields.io/badge/style-ePic%20Deep%20Space-080c14.svg)]()

**ePic** is a high-performance, modular **Particle-in-Cell (PIC)** plasma simulation engine written in vectorized Python. It resolves the collisionless kinetic dynamics of charged particles governed by the coupled **Vlasov–Poisson** and **Vlasov–Maxwell** systems across 1D-3V, 2D-3V, and 3D-3V geometries.

Originally developed at the **Department of Astronomy, Astrophysics and Space Engineering, Indian Institute of Technology Indore (IIT Indore)** in conjunction with research on magnetic reconnection and plasmoid instability (*M.Sc. Thesis, Avinash Kumar Himanshu, 2023, under Dr. Bhargav Vaidya*), `ePic` has been re-architected into a production-grade scientific computing suite with a unified deep-space visualization theme, exact symplectic leapfrog integration, and zero numerical self-force.

---

## Interactive Google Colab Notebooks

Explore, execute, and modify all plasma physics benchmarks in your browser with zero installation:

| Notebook | Focus & Physical Regime | 1-Click Launch |
| :--- | :--- | :---: |
| **01. Boris Pusher Verification** | Leapfrog time-staggering, cyclotron orbits, $E \times B$ drift, machine-precision energy conservation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/01_Boris_Pusher_Verification.ipynb) |
| **02. 1D-3V Two-Stream Instability** | Penrose criterion, linear growth rate, BGK phase-space vortex roll-up, coarse-grained entropy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/02_1D_Two_Stream_Instability.ipynb) |
| **03. 1D-3V Landau Damping** | Collisionless phase mixing, Bohm-Gross oscillation, analytic Landau decay rate $\gamma_L$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/03_1D_Landau_Damping.ipynb) |
| **04. 2D PIC Filamentation & Reconnection** | Oblique wave modes, current channels, Harris sheet tearing mode, plasmoid formation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/04_2D_PIC_Filamentation_and_Reconnection.ipynb) |
| **05. 3D PIC Plasma Expansion** | Spherical Coulomb explosion, ambipolar electric fields, self-similar expansion fronts | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/05_3D_PIC_Plasma_Expansion.ipynb) |
| **06. Asymmetric Reconnection (M.Sc. Thesis)** | Dayside magnetopause, Cassak-Shay scaling, X-line & stagnation decoupling, Hall $B_z$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/06_Asymmetric_Reconnection_MSc_Thesis.ipynb) |
| **07. Kinetic Waves & Shocks** | First-principles $(k, \omega)$ dispersion reconstruction, supersonic shock ramp, ion reflection foot | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/07_Kinetic_Plasma_Waves_and_Shocks.ipynb) |
| **Master Tutorial Notebook** | Comprehensive end-to-end guide across the entire ePic architecture and API | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avinash-tiwary/ePic/blob/main/notebooks/ePic_Master_Tutorial.ipynb) |

---

## Animated Simulation Movies

All movies are rendered directly from kinetic particle coordinates using the unified `ePic` deep-space styling engine (`#080c14` canvas):

| 1D Two-Stream Phase Space Vortex | 1D Collisionless Landau Damping |
| :---: | :---: |
| ![Two-Stream 1D](docs/animations/two_stream_1d.gif) | ![Landau Damping 1D](docs/animations/landau_damping_1d.gif) |
| *Phase-space vortex roll-up, velocity distribution flattening, and field energy growth marker* | *Phase-mixing of kinetic electrons and exponential damping of wave potential* |

| 2D Asymmetric Magnetic Reconnection | 1D Collisionless Electrostatic Shock |
| :---: | :---: |
| ![Asymmetric Reconnection](docs/animations/asymmetric_reconnection_2d.gif) | ![Electrostatic Shock](docs/animations/electrostatic_shock_1d.gif) |
| *Dayside magnetopause current sheet thinning, X-point decoupling, and magnetic island dynamics* | *Supersonic collision ($M=2.0$), steep electrostatic barrier, and specular ion reflection foot* |

| 2D Transverse Beam Filamentation | 3D Spherical Coulomb Explosion |
| :---: | :---: |
| ![Filamentation 2D](docs/animations/filamentation_2d.gif) | ![Plasma Expansion 3D](docs/animations/plasma_expansion_3d.gif) |
| *Weibel-type current filamentation and transverse density clustering* | *3D rotating volume scatter of expanding kinetic ions accelerated by ambipolar fields* |

---

## Benchmark Physics Dashboards

Every dashboard adheres to the strict **ePic Unified Visual Standard** (`src/epic/diagnostics/style.py`): cosmic obsidian canvas (`#080c14`), twilight navy axes (`#0d1322`), glowing plasma colormaps, and crisp typography.

### 1. 1D-3V Two-Stream Instability
Simulates counter-streaming electron beams ($v = \pm 3.0\,v_{th}$), capturing linear exponential growth, nonlinear harmonic cascades, and exact energy conservation.

| (a) Phase-Space Vortex Roll-Up | (b) Energy Partition & Modal Cascades |
| :---: | :---: |
| ![Two-Stream Phase Space](docs/images/two_stream_phase_space.png) | ![Two-Stream Energy](docs/images/two_stream_energy.png) |

- **Linear Growth Validation**: Measured numerical growth rate $\gamma \approx 0.181\,\omega_{pe}$ agrees with warm-beam linear dispersion theory.
- **Symplectic Conservation**: Relative total energy drift $\Delta E / E_0 = 7.5 \times 10^{-5}$ ($< 0.008\%$) over 500 timesteps.

---

### 2. 1D-3V Collisionless Landau Damping & Bohm-Gross Resonance
Langmuir wave damping without physical collisions via resonant wave-particle phase mixing.

<p align="center">
  <img src="docs/images/landau_damping.png" width="90%" alt="Landau Damping Dashboard" />
</p>

- **Analytic Rate**:
  $$\gamma_L = -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3} \exp\left(-\frac{1}{2(k\lambda_D)^2} - \frac{3}{2}\right) \approx -0.1514\,\omega_{pe}$$
- The measured peak electric field envelope $|E|_{\max}(t)$ follows the analytic linear Landau decay rate down to the kinetic thermal noise floor.

---

### 3. 1D-3V Non-linear Landau Damping & O'Neil Trapping Oscillations
When the wave amplitude is large ($\alpha = 0.45$), resonant electrons become trapped in the electrostatic potential troughs, producing characteristic O'Neil bounce oscillations ($\tau_B \approx 13.2\,\omega_{pe}^{-1}$).

<p align="center">
  <img src="docs/images/nonlinear_landau_trapping.png" width="90%" alt="Nonlinear Landau Trapping" />
</p>

---

### 4. 1D-3V Bump-on-Tail Quasilinear Relaxation
A gentle high-energy electron beam ($v_b = 4.5\,v_{th}$) excites waves via inverse Landau damping ($\partial f/\partial v > 0$), diffusing resonant electrons into a flat quasilinear velocity plateau ($\partial f/\partial v = 0$).

<p align="center">
  <img src="docs/images/bump_on_tail_plateau.png" width="90%" alt="Bump on Tail Plateau" />
</p>

---

### 5. First-Principles $(k, \omega)$ Wave Dispersion Reconstruction
By performing a 2D spatio-temporal Fourier transform of thermal electric fluctuations $E(x, t)$, the exact kinetic dispersion surface is illuminated, directly verifying the **Bohm-Gross Langmuir relation**:
$$\omega^2(k) = \omega_{pe}^2 + 3 k^2 v_{th}^2$$

<p align="center">
  <img src="docs/images/plasma_dispersion_relation.png" width="90%" alt="Plasma Dispersion Relation" />
</p>

---

### 6. 1D-3V Collisionless Electrostatic Shock Wave
Supersonic collision ($M = 2.0$) of two plasma slabs generates an electrostatic shock barrier $\Delta \phi$ that specularly reflects incoming ions into a characteristic phase-space "shock foot" ($v_{\text{ref}} = 2 v_s - v_{\text{in}}$).

<p align="center">
  <img src="docs/images/electrostatic_shock_dynamics.png" width="90%" alt="Electrostatic Shock Wave" />
</p>

---

### 7. 2D-3V Beam Filamentation & Spatial Current Channels
Resolves 2D transverse filamentation (Weibel-type clustering) and oblique wave coupling alongside longitudinal phase-space vortex coalescence.

<p align="center">
  <img src="docs/images/2d_filamentation.png" width="90%" alt="2D Filamentation Instability" />
</p>

---

### 8. 2D-3V Harris Sheet Magnetic Reconnection & Plasmoids
Equilibrium current sheet separating anti-parallel magnetic fields:
$$\mathbf{B}(y) = B_0 \tanh\left(\frac{y - y_c}{L_p}\right) \hat{\mathbf{x}} + \delta \mathbf{B}(x, y)$$

<p align="center">
  <img src="docs/images/harris_reconnection.png" width="90%" alt="Harris Sheet Reconnection" />
</p>

---

### 9. 2D-3V Asymmetric Dayside Magnetopause Reconnection
Directly models the astrophysical space plasma regime investigated in the author's IIT Indore Master's thesis (*Avinash Kumar Himanshu, 2023, under Dr. Bhargav Vaidya*):

<p align="center">
  <img src="docs/images/asymmetric_reconnection_dashboard.png" width="90%" alt="Asymmetric Dayside Reconnection" />
</p>

- **Asymmetric Boundary Layer**: Magnetosphere ($B_1=1.0, n_1=0.6$) vs Magnetosheath ($B_2=0.5, n_2=2.0$).
- **Decoupled Nulls**: Magnetic $X$-point and hydrodynamic flow stagnation point drift apart spatially.
- **Non-Thermal Power-Law Acceleration**: Clear power-law spectrum $dN/dE \propto E^{-3.2}$ of electrons escaping along the magnetic separatrix.

---

### 10. 3D-3V Spherical Plasma Expansion & Coulomb Explosion
3D electrostatic simulation of a localized plasma cloud expanding into vacuum under self-consistent ambipolar space-charge forces.

<p align="center">
  <img src="docs/images/3d_plasma_expansion.png" width="90%" alt="3D Spherical Plasma Expansion" />
</p>

---

## Theoretical Foundations

In-depth mathematical derivations and numerical formulations are available in `docs/theory/`:

1. [**01. PIC Fundamentals & Vlasov-Poisson Systems**](docs/theory/01_pic_fundamentals.md): Macroparticle weighting, normalization scales ($\omega_{pe}, \lambda_D, v_{th}$), and stability constraints ($\omega_{pe} \Delta t \le 0.1$).
2. [**02. The Symplectic Boris Algorithm**](docs/theory/02_boris_algorithm.md): Exact rotation derivation, leapfrog velocity staggering, and machine-precision magnetic invariants.
3. [**03. Field Solvers & Cloud-In-Cell Interpolation**](docs/theory/03_field_solvers_and_cic.md): Spectral discrete Laplacian eigenvalues ($k_{\text{eff}}^2$), dual adjoint interpolation, and zero self-force proof.
4. [**04. Kinetic Magnetic Reconnection & Harris Sheets**](docs/theory/04_magnetic_reconnection.md): Harris equilibrium pressure balance, plasmoid instability, and Cassak-Shay asymmetric reconnection scaling.

---

## Core Computational Engine

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
Solves the Lorentz equation:
$$\frac{d\mathbf{v}}{dt} = \frac{q}{m} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)$$
- **Symplectic Leapfrog Time-Centering**: Velocity is retarded by $-\Delta t/2$ at $t=0$, guaranteeing exact second-order global convergence $\mathcal{O}(\Delta t^2)$.
- **Exact Scalar Denominator**: Computes $1 + |\mathbf{t}|^2 = 1 + \mathbf{t}\cdot\mathbf{t}$ as a true scalar, preserving the magnetic invariant $|\mathbf{v}| = \text{const}$ to machine precision ($< 10^{-12}$).

### 2. Spectral (FFT) Poisson Solver
- Solves $\nabla^2 \phi = -\rho / \varepsilon_0$ in $\mathcal{O}(N \log N)$ operations via Fast Fourier Transforms.
- Uses exact finite-difference discrete Laplacian eigenvalues:
  $$k_{\text{eff}}^2 = \left(\frac{2}{\Delta x}\sin\frac{k_x \Delta x}{2}\right)^2 + \left(\frac{2}{\Delta y}\sin\frac{k_y \Delta y}{2}\right)^2 + \left(\frac{2}{\Delta z}\sin\frac{k_z \Delta z}{2}\right)^2$$
- Gauge-invariant: strictly sets $\hat{\phi}(0) = 0$ to prevent numerical potential drift.

### 3. Volume-Conserving Cloud-in-Cell (CIC)
- Implemented via vectorized index histograms (`np.bincount` / `np.add.at`).
- Dual adjoint deposition and interpolation kernels guarantee **exact momentum conservation and zero numerical self-force**.

---

## Installation & Setup

Clone the repository and set up your Python environment:

```bash
git clone https://github.com/avinash-tiwary/ePic.git
cd ePic

# Using standard venv + pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Or using uv (recommended for 10x faster installation)
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Running the Benchmark Experiments

Run any of the 10 physics experiments with one command:

```bash
# 1D Two-stream instability (phase space vortex & harmonic cascades)
python experiments/1d_two_stream.py

# 1D Collisionless Landau damping (analytic gamma_L envelope)
python experiments/1d_landau_damping.py

# 1D Non-linear Landau damping & O'Neil bounce oscillations
python experiments/1d_nonlinear_landau.py

# 1D Bump-on-tail quasilinear plateau relaxation
python experiments/1d_bump_on_tail.py

# 1D First-principles (k, omega) plasma dispersion relation
python experiments/1d_plasma_dispersion.py

# 1D Supersonic collisionless electrostatic shock wave
python experiments/1d_electrostatic_shock.py

# 2D Two-stream & transverse filamentation instability
python experiments/2d_two_stream.py

# 2D Symmetric Harris current sheet reconnection
python experiments/2d_harris_reconnection.py

# 2D Asymmetric dayside magnetopause reconnection (M.Sc. thesis)
python experiments/2d_asymmetric_reconnection.py

# 3D Spherical plasma expansion & Coulomb explosion
python experiments/3d_plasma_expansion.py

# Generate all 7 animated movies/GIFs
python experiments/generate_movies.py
```

---

## Automated Unit Tests

Run the full 13-test automated test suite:

```bash
pytest tests/
```

```
============================= test session starts ==============================
collected 13 items

tests/test_1d_pic.py .                                                   [  7%]
tests/test_2d_pic.py .                                                   [ 15%]
tests/test_3d_pic.py .                                                   [ 23%]
tests/test_boris.py ....                                                 [ 53%]
tests/test_cic.py ...                                                    [ 76%]
tests/test_poisson.py ...                                                [100%]

============================== 13 passed in 1.95s ==============================
```

---

## Scientific Bibliography

```bibtex
@article{boris1970relativistic,
  author  = {Boris, J. P.},
  title   = {Relativistic plasma simulation-optimization of a hybrid code},
  journal = {Proc. Fourth Conf. Num. Sim. Plasmas},
  pages   = {3--67},
  year    = {1970},
  publisher = {Naval Res. Lab, Wash. D.C.}
}

@book{birdsall2004plasma,
  author    = {Birdsall, C. K. and Langdon, A. B.},
  title     = {Plasma Physics via Computer Simulation},
  publisher = {CRC Press / Taylor & Francis},
  year      = {2004}
}

@book{hockney1988computer,
  author    = {Hockney, R. W. and Eastwood, J. W.},
  title     = {Computer Simulation Using Particles},
  publisher = {Adam Hilger, Bristol},
  year      = {1988}
}

@mastersthesis{himanshu2023reconnection,
  author  = {Himanshu, Avinash Kumar},
  title   = {Characterising Magnetic Reconnection in Asymmetric Medium},
  school  = {Department of Astronomy, Astrophysics and Space Engineering, Indian Institute of Technology Indore},
  year    = {2023},
  note    = {Advisor: Dr. Bhargav Vaidya}
}

@article{cassak2007scaling,
  author  = {Cassak, P. A. and Shay, M. A.},
  title   = {Scaling of asymmetric magnetic reconnection: General theory and collisional simulations},
  journal = {Physics of Plasmas},
  volume  = {14},
  number  = {10},
  pages   = {102114},
  year    = {2007}
}

@article{landau1946vibrations,
  author  = {Landau, L. D.},
  title   = {On the vibrations of the electronic plasma},
  journal = {Journal of Physics (USSR)},
  volume  = {10},
  pages   = {25},
  year    = {1946}
}

@article{oneil1965collisionless,
  author  = {O'Neil, T. M.},
  title   = {Collisionless damping of nonlinear plasma oscillations},
  journal = {The Physics of Fluids},
  volume  = {8},
  number  = {12},
  pages   = {2255--2262},
  year    = {1965}
}

@article{forslund1971formation,
  author  = {Forslund, D. W. and Freidberg, J. P.},
  title   = {Theory of Laminar Collisionless Electrostatic Shocks},
  journal = {Physical Review Letters},
  volume  = {27},
  number  = {18},
  pages   = {1189},
  year    = {1971}
}

@article{derouillat2018smilei,
  author  = {Derouillat, J. and others},
  title   = {SMILEI: A collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation},
  journal = {Computer Physics Communications},
  volume  = {222},
  pages   = {351--373},
  year    = {2018}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
