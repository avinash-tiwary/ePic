# 1. Fundamentals of Particle-in-Cell (PIC) Simulation

## 1. The Kinetic Vlasov–Poisson System
A collisionless, unmagnetized (or weakly magnetized) plasma of charged species $s$ (electrons, ions) is rigorously described in 6D phase space $(\mathbf{x}, \mathbf{v})$ by the **Vlasov equation**:
$$\frac{\partial f_s}{\partial t} + \mathbf{v} \cdot \nabla_{\mathbf{x}} f_s + \frac{q_s}{m_s} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right) \cdot \nabla_{\mathbf{v}} f_s = 0$$

where $f_s(\mathbf{x}, \mathbf{v}, t)$ is the single-particle distribution function representing the probability density of finding a particle of species $s$ at position $\mathbf{x}$ with velocity $\mathbf{v}$ at time $t$.

In electrostatic limits, the magnetic field is either external or negligible, and the self-consistent electric field $\mathbf{E} = -\nabla \phi$ satisfies **Poisson’s equation**:
$$\nabla^2 \phi(\mathbf{x}, t) = -\frac{\rho(\mathbf{x}, t)}{\epsilon_0} = -\frac{1}{\epsilon_0} \sum_s q_s \int f_s(\mathbf{x}, \mathbf{v}, t) \, d^3\mathbf{v}$$

---

## 2. Macroparticle Discretization
Because physical plasmas contain astronomical particle counts ($n_e \sim 10^{18} - 10^{26}\,\text{m}^{-3}$), directly tracking every real electron is computationally impossible. In the PIC method, the continuous distribution function $f_s$ is discretized into an ensemble of $N_p$ **macroparticles** (or superparticles):
$$f_s(\mathbf{x}, \mathbf{v}, t) \approx \sum_{i=1}^{N_p} w_i \, S\left(\mathbf{x} - \mathbf{x}_i(t)\right) \, \delta\left(\mathbf{v} - \mathbf{v}_i(t)\right)$$
where:
- $w_i = \frac{n_0 V}{N_p}$ is the particle numerical weight (representing the number of real physical particles encapsulated in one macroparticle).
- $S(\mathbf{x})$ is the finite-size particle shape function (Cloud-in-Cell b-spline).
- Each macroparticle maintains the physical charge-to-mass ratio:
  $$\left(\frac{q}{m}\right)_{\text{macro}} = \left(\frac{q}{m}\right)_{\text{real}}$$

---

## 3. Characteristic Plasma Scales & Numerical Stability

To ensure that the simulation captures true kinetic physics without exciting artificial numerical instabilities, three fundamental criteria must be strictly satisfied:

### 1. Plasma Frequency & Time-Step Criterion
The fundamental timescale of electron oscillations is the **electron plasma frequency**:
$$\omega_{pe} = \sqrt{\frac{n_0 q_e^2}{\epsilon_0 m_e}}$$
For the second-order leapfrog integrator to remain numerically stable and avoid artificial grid heating, the time step must satisfy:
$$\omega_{pe} \Delta t \le 0.1 - 0.2$$
If $\omega_{pe} \Delta t > 2$, the leapfrog amplification factor exceeds unity ($|\lambda| > 1$), causing catastrophic exponential numerical explosion.

### 2. Debye Length & Grid Resolution
The length scale over which mobile charges screen out electric fields is the **electron Debye length**:
$$\lambda_D = \sqrt{\frac{\epsilon_0 k_B T_e}{n_0 q_e^2}} = \frac{v_{th, e}}{\omega_{pe}}$$
To prevent **numerical finite-grid instability** (where non-physical aliases transfer energy from the mesh into particle thermal motion), the grid spacing $\Delta x$ must resolve the Debye length:
$$\Delta x \lesssim (1 - 3) \, \lambda_D$$

### 3. Courant-Friedrichs-Lewy (CFL) Condition
Particles must not traverse more than one grid cell in a single time step:
$$v_{\max} \Delta t < \Delta x \implies \frac{v_{\max} \Delta t}{\Delta x} < 1$$
