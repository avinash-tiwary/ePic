import numpy as np
import matplotlib.pyplot as plt

# Particle pusher
def borisPush(v_in, E, B, q, m, dt):

    t = (q*B/m)*(dt/2)
    s = (2*t)/(1 + t**2)

    v_minus = v_in + (q*E/m)*dt/2
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_out = v_plus + (q*E/m)*dt/2

    return v_out


N = 100
t = np.linspace(0, 10, N)
dt = t[1] - t[0]

x = np.zeros([N, 3])
v = np.zeros([N, 3])
K = np.zeros(N)

E = np.array([0, 0, 0])
B = np.array([0, 0, 10])
m = 1
q = 1
v_in = np.array([1, 0, 0])

v[0] = v_in
for i in range(N-1):
    v[i+1] = borisPush(v[i], E, B, q, m, dt)
    x[i+1] = x[i] + v[i+1]*dt
    K[i] = np.linalg.norm(v[i])


# Plotting the particle's trajectory
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x[:,0], x[:,1], t, 'green')
plt.show()
