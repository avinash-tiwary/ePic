# 2. The Symplectic Boris Particle Pusher

## 1. The Physical Problem
The motion of a charged particle in arbitrary, inhomogeneous electromagnetic fields $\mathbf{E}(\mathbf{x})$ and $\mathbf{B}(\mathbf{x})$ is governed by the non-relativistic Lorentz force:
$$\frac{d\mathbf{x}}{dt} = \mathbf{v}$$
$$m \frac{d\mathbf{v}}{dt} = q \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)$$

A critical property of the Lorentz force is that **the magnetic field does no work**:
$$\frac{d}{dt}\left(\frac{1}{2}m |\mathbf{v}|^2\right) = q \, \mathbf{v} \cdot (\mathbf{E} + \mathbf{v} \times \mathbf{B}) = q \, \mathbf{v} \cdot \mathbf{E}$$
In a pure magnetic field ($\mathbf{E} = 0$), particle kinetic energy is an **exact invariant of motion**:
$$|\mathbf{v}(t)| = \text{const}$$

Standard Runge–Kutta integrators (RK2, RK4) fail this condition: over thousands of gyro-orbits, they cause particles to artificially spiral inward or outward due to numerical dissipation.

---

## 2. The Boris Scheme (Boris, 1970)
The Boris algorithm separates the electric field acceleration from the magnetic field rotation. The time step $\Delta t$ is split into three symmetric stages:

```
v^{n-1/2} ───[ + (qE/m)*dt/2 ]───> v^- ───[ Pure Magnetic Rotation ]───> v^+ ───[ + (qE/m)*dt/2 ]───> v^{n+1/2}
```

### Stage 1: Half-Step Electric Acceleration
$$\mathbf{v}^- = \mathbf{v}^{n-1/2} + \frac{q \mathbf{E}^n}{m} \frac{\Delta t}{2}$$

### Stage 2: Pure Magnetic Rotation
We define the dimensionless rotation vector $\mathbf{t}$:
$$\mathbf{t} \equiv \frac{q \mathbf{B}^n}{m} \frac{\Delta t}{2}$$
And the modified rotation vector $\mathbf{s}$:
$$\mathbf{s} \equiv \frac{2\mathbf{t}}{1 + |\mathbf{t}|^2}$$

The vector $\mathbf{v}^-$ is rotated by angle $\theta \approx -2\arctan(|\mathbf{t}|)$ around $\mathbf{B}$:
1. Half-rotation auxiliary vector:
   $$\mathbf{v}' = \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t}$$
2. Full rotation vector:
   $$\mathbf{v}^+ = \mathbf{v}^- + \mathbf{v}' \times \mathbf{s}$$

### Proof of Exact Energy Preservation in Magnetic Rotation
$$\mathbf{v}^+ - \mathbf{v}^- = \mathbf{v}' \times \mathbf{s}$$
Taking the dot product with $\mathbf{v}' = \mathbf{v}^- + \frac{1}{2}(\mathbf{v}^+ - \mathbf{v}^-)$:
$$(\mathbf{v}^+ - \mathbf{v}^-) \cdot (\mathbf{v}^+ + \mathbf{v}^-) = 2 \, (\mathbf{v}' \times \mathbf{s}) \cdot \mathbf{v}' \equiv 0$$
$$|\mathbf{v}^+|^2 - |\mathbf{v}^-|^2 = 0 \implies |\mathbf{v}^+| = |\mathbf{v}^-| \quad \text{(Exact to machine precision!)}$$

### Stage 3: Second Half-Step Electric Acceleration
$$\mathbf{v}^{n+1/2} = \mathbf{v}^+ + \frac{q \mathbf{E}^n}{m} \frac{\Delta t}{2}$$

---

## 3. Position Drift & Leapfrog Staggering
Once $\mathbf{v}^{n+1/2}$ is obtained, the positions are updated via:
$$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1/2} \Delta t$$

### Velocity Retardation at $t = 0$
To initialize the leapfrog cycle from physical initial conditions $\mathbf{x}(t=0), \mathbf{v}(t=0)$, the velocity is retarded to $t = -\Delta t / 2$:
$$\mathbf{v}^{-1/2} = \text{BorisPush}\left(\mathbf{v}^0, \mathbf{E}^0, \mathbf{B}^0, q, m, -\frac{\Delta t}{2}\right)$$
This ensures the entire integration remains globally **second-order accurate $\mathcal{O}(\Delta t^2)$ and symplectic**.
