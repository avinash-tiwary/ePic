import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def acceleration(pos, Nx, boxsize, n0, gradientMatrix, A_matrix, q, m):

    N = pos.shape[0]
    dx = boxsize / Nx

    # ****************************************************************
    # CIC
    # ////////////////////////////////////////////////////////////////
    i = np.floor(pos/dx).astype(int)

    ip1 = i+1  
    weight_i = (ip1*dx - pos)/dx
    weight_ip1 = (pos - i*dx)/dx
    ip1 = np.mod(ip1, Nx)   # periodic BC

    n = np.bincount(i[:, 0],   weights=weight_i[:, 0],   minlength=Nx)
    n += np.bincount(ip1[:, 0], weights=weight_ip1[:, 0], minlength=Nx)

    n *= n0 * boxsize / N / dx
    # /////////////////////////////////////////////////////////////////
    # ****************************************************************


    # Solve Poisson's Equation (using finit difference method)
    phi_grid = spsolve(A_matrix, n-n0, permc_spec="MMD_AT_PLUS_A")

    # Finding gradient get the Electric field (E = -grad(phi))
    E_grid = - gradientMatrix @ phi_grid
    # E_grid = np.gradient(phi_grid, 2)

    # Interpolate grid value onto particle locations (Inverse CIC)
    E = weight_i * E_grid[i] + weight_ip1 * E_grid[ip1]
    a = -q*E/m

    return a


# //////////////////////////////////////////////////////////////
# Input parameters
N = 100000     # Number of particles
Nx = 400       # Number of mesh cells
t = 0          # current time of the simulation
tEnd = 50      # time at which simulation ends
dt = 1         # timestep
boxsize = 45   # Size of Simulation area
n0 = 1         # electron number density
m = 1          # Mass of particle (taken unity here)
q = 1		   # Charge on particle
vb = 3         # beam velocity
vsp = 1        # Beam Velocity spread
# //////////////////////////////////////////////////////////////

# Constructing two beams moving in opposite direction
pos = np.random.rand(N, 1) * boxsize
vel = vsp * np.random.randn(N, 1) + vb
Nh = int(N/2)
vel[Nh:] *= -1


#////////////////////////////////////////////////////////////////
# Construct tridiagonal matrix A (Ax=b) to solve poisson Equation
# Using Finite difference method (Elliptic PDE)
dx = boxsize/Nx
e = np.ones(Nx)
diags = np.array([-1, 0, 1])
vals = np.vstack((e, -2*e, e))
A_matrix = sp.spdiags(vals, diags, Nx, Nx)
A_matrix = sp.lil_matrix(A_matrix)
A_matrix[0, Nx-1] = 1
A_matrix[Nx-1, 0] = 1
A_matrix /= dx**2
A_matrix = sp.csr_matrix(A_matrix)

# Construct Gradient matrix to find geadient of electric potential 
# to find Electric field (E = -grad(phi))
diags = np.array([-1, 1])
vals = np.vstack((-e, e))
gradientMatrix = sp.spdiags(vals, diags, Nx, Nx)
gradientMatrix = sp.lil_matrix(gradientMatrix)
gradientMatrix[0, Nx-1] = -1
gradientMatrix[Nx-1, 0] = 1
gradientMatrix /= (2*dx)
gradientMatrix = sp.csr_matrix(gradientMatrix)
#////////////////////////////////////////////////////////////////
# Initializing the PIC simulation
#////////////////////////////////////////////////////////////////
# Initial force calculation on particle for first kick
acc = acceleration(pos, Nx, boxsize, n0, gradientMatrix, A_matrix, q, m)
fig = plt.figure(figsize=(10, 6), dpi=100)
# # number of timesteps
Nt = int(tEnd/dt)

#////////////////////////////////////////////////////////////////
# Main PIC loop:
for i in range(Nt):

    # (1/2) kick
    vel += acc * dt/2.0
    # drift (With periodic boundary conditions)
    pos += vel * dt
    pos = np.mod(pos, boxsize)
    # Acceleration on this new point
    acc = acceleration(pos, Nx, boxsize, n0, gradientMatrix, A_matrix, q, m)
    # (1/2) kick
    vel += acc * dt/2.0
    # updating time
    t += dt

    plt.cla()
    plt.scatter(pos[0:Nh], vel[0:Nh], s=.4, color='blue', alpha=0.5)
    plt.scatter(pos[Nh:], vel[Nh:], s=0.4, color='red',  alpha=0.5)
    plt.axis([0, boxsize, -6, 6])
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title(f"{np.round(t,2)}/{tEnd}")
    plt.savefig(f'{t}.png')
    plt.pause(0.001)
    

#////////////////////////////////////////////////////////////////
# It Ends here
#////////////////////////////////////////////////////////////////
