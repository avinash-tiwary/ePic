# 4. Kinetic Magnetic Reconnection & Harris Current Sheets

## 1. Physical Significance
Magnetic reconnection is a fundamental plasma process wherein magnetic topology is rapidly rearranged, converting stored magnetic field energy into plasma thermal energy, bulk kinetic outflow, and non-thermal particle acceleration.

It drives explosive phenomena across astrophysics and space physics:
- Solar flares and Coronal Mass Ejections (CMEs).
- Earth's magnetospheric substorms (magnetopause and magnetotail reconnection).
- Astrophysical relativistic jets, accretion disks, and gamma-ray bursts (GRBs).

---

## 2. The 1D Harris Current Sheet Equilibrium
In 1962, E. G. Harris derived the exact kinetic equilibrium for a collisionless neutral plasma sheet separating two regions of anti-parallel magnetic fields:
$$\mathbf{B}_0(y) = B_0 \tanh\left(\frac{y - y_c}{L_p}\right) \hat{\mathbf{x}}$$
where:
- $L_p$ is the current sheet half-thickness.
- $y_c$ is the center of the current sheet.

### Pressure Balance
Ampère’s law dictates the out-of-plane current density $J_z$:
$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} \implies J_z(y) = -\frac{d B_x}{dy} = -\frac{B_0}{L_p} \operatorname{sech}^2\left(\frac{y - y_c}{L_p}\right)$$
Total pressure balance (thermal pressure + magnetic pressure = const) requires:
$$P(y) + \frac{B^2(y)}{2\mu_0} = \text{const}$$
$$n(y) = n_0 \operatorname{sech}^2\left(\frac{y - y_c}{L_p}\right) + n_b$$
where $n_b$ is the background asymptotic lobe density.

---

## 3. Kinetic Reconnection & Plasmoid Instability
When a perturbation $\delta A_z(x, y)$ is applied:
$$\delta \mathbf{B} = \nabla \times (\delta A_z \hat{\mathbf{z}}) = \frac{\partial \delta A_z}{\partial y} \hat{\mathbf{x}} - \frac{\partial \delta A_z}{\partial x} \hat{\mathbf{y}}$$
the continuous magnetic sheet breaks up via the **tearing mode instability**, forming:
- **X-points**: Regions of vanishing magnetic field where field lines reconnect and decouple from the electron fluid.
- **O-points (Magnetic Islands / Plasmoids)**: Closed magnetic loops that trap high-energy plasma and are accelerated outward by magnetic tension forces (outflow jets).

### Connection to Thesis Research (IIT Indore, 2023)
In realistic space environments (e.g. Earth's dayside magnetopause), reconnection is **asymmetric**: the density, magnetic field strength, and plasma beta differ drastically on either side of the diffusion region ($B_1 \neq B_2, \rho_1 \neq \rho_2$). This causes the X-line and stagnation point to decouple and drift, altering the reconnection rate:
$$R \sim \frac{v_{in}}{v_{out}} \approx 0.1$$
`ePic` provides the kinetic building blocks to model asymmetric inflows and verify kinetic Hall reconnection against large-scale codes like SMILEI.
