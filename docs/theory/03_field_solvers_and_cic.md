# 3. Field Solvers & Cloud-in-Cell (CIC) Weighting

## 1. Cloud-in-Cell (CIC) Charge Deposition
The Cloud-in-Cell scheme models each particle not as an infinitely sharp Dirac delta function $\delta(\mathbf{x})$, but as a uniform cloud or box of dimension $\Delta x \times \Delta y \times \Delta z$.

In 1D, for a particle located at position $x_p \in [x_i, x_{i+1}]$:
$$\xi = \frac{x_p}{\Delta x}, \quad i = \lfloor \xi \rfloor, \quad f = \xi - i$$
The charge $q_p$ is apportioned between the adjacent grid vertices $i$ and $i+1$ via linear volume weighting:
$$w_i = 1 - f = \frac{x_{i+1} - x_p}{\Delta x}$$
$$w_{i+1} = f = \frac{x_p - x_i}{\Delta x}$$
The charge density at grid node $j$ is the sum over all particles:
$$\rho_j = \frac{1}{\Delta x} \sum_{p=1}^{N_p} q_p \, W(x_j - x_p)$$

---

## 2. Zero Self-Force Theorem
A critical requirement of any physical PIC code is that **an isolated particle must feel zero net force from its own deposited charge cloud**:
$$\mathbf{F}_{\text{self}} = q_p \, \mathbf{E}_{\text{self}}(x_p) \equiv 0$$
If the field interpolation scheme differs from the charge deposition kernel, a particle exerts a non-zero fictitious self-force on itself, resulting in artificial self-acceleration and non-conservation of global momentum.

By choosing the electric field interpolation to be the **exact adjoint** of the deposition operator:
$$E(x_p) = (1 - f) E_i + f E_{i+1}$$
momentum conservation $\sum_i \mathbf{F}_i = 0$ is guaranteed to machine precision.

---

## 3. Spectral (FFT) Poisson Solver
Poisson's equation on a uniform periodic grid of size $L$ is:
$$\frac{d^2 \phi}{dx^2} = -\frac{\rho(x) - \bar{\rho}}{\epsilon_0}$$
where $\bar{\rho} = \frac{1}{L} \int_0^L \rho(x) dx$ enforces global charge neutrality.

Taking the discrete Fourier transform:
$$\phi(x) = \sum_k \hat{\phi}_k e^{i k x}, \quad \rho(x) = \sum_k \hat{\rho}_k e^{i k x}$$

### Modified Wave Number Formulation
On a discrete grid, the central-difference Laplacian is:
$$\frac{\phi_{j+1} - 2\phi_j + \phi_{j-1}}{\Delta x^2} \longleftrightarrow -k_{\text{eff}}^2 \hat{\phi}_k$$
where the **exact modified discrete wave number** is:
$$k_{\text{eff}}^2 = \left( \frac{2}{\Delta x} \sin\left(\frac{k \Delta x}{2}\right) \right)^2$$
Inverting in Fourier space:
$$\hat{\phi}_k = \begin{cases} 
\frac{\hat{\rho}_k}{\epsilon_0 k_{\text{eff}}^2} & \text{for } k \neq 0 \\
0 & \text{for } k = 0 \quad \text{(Gauge condition)}
\end{cases}$$

The electric field in Fourier space is:
$$\hat{E}_k = -i \left[ \frac{\sin(k \Delta x)}{\Delta x} \right] \hat{\phi}_k$$
Transforming back via inverse FFT gives the exact, smooth, self-force-free electric field $\mathbf{E}(\mathbf{x})$ in $\mathcal{O}(N \log N)$ operations.
