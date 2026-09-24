"""
Master Animation Generator for ePic Suite
=========================================
Generates high-definition scientific movies and GIFs for:
1. 1D Two-Stream Instability (Phase space vortex roll-up + distribution function)
2. 1D Collisionless Landau Damping (Wave phase-mixing)
3. 2D Two-Stream & Filamentation Instability (2D density evolution)
4. 2D Harris Sheet Magnetic Reconnection (Dynamic magnetic islands & X-points)
5. 3D Spherical Plasma Expansion (Rotating 3D scatter)

Run:
    python experiments/generate_movies.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from epic.solvers.pic1d import PIC1DSolver
from epic.solvers.pic2d import PIC2DSolver
from epic.solvers.pic3d import PIC3DSolver
from epic.field.cic import deposit_charge_1d, interpolate_field_1d, deposit_charge_2d, interpolate_field_2d
from epic.field.poisson import solve_poisson_1d_fft, solve_poisson_2d_fft
from epic.pusher.boris import boris_push, retard_velocity
from epic.diagnostics.visualization import animate_1d_phase_space, animate_2d_density
from epic.diagnostics.style import apply_epic_style, EPIC_COLORS


def generate_two_stream_movie():
    print("\n[1/5] Generating 1D Two-Stream Instability Movie...")
    Nx = 256
    boxsize = 45.0
    N_particles = 60000
    dt = 0.1
    t_end = 45.0

    weight = (1.0 * boxsize) / N_particles
    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    Nh = N_particles // 2
    np.random.seed(42)
    pos1 = np.random.uniform(0.0, boxsize, Nh)
    vel1 = np.random.normal(3.0, 0.5, (Nh, 3))
    pos2 = np.random.uniform(0.0, boxsize, Nh)
    vel2 = np.random.normal(-3.0, 0.5, (Nh, 3))

    solver.add_species("beam1", q=-weight, m=weight, pos=pos1, vel=vel1)
    solver.add_species("beam2", q=-weight, m=weight, pos=pos2, vel=vel2)
    solver.initialize()

    snapshots = []
    def save_snap(s):
        snapshots.append({
            "time": s.time,
            "beam1": (s.species[0].pos[:, 0].copy(), s.species[0].vel[:, 0].copy()),
            "beam2": (s.species[1].pos[:, 0].copy(), s.species[1].vel[:, 0].copy()),
        })

    save_snap(solver)
    solver.run(t_end=t_end, callback=save_snap, callback_interval=5)

    time_hist = np.array(solver.history["time"])
    field_hist = np.array(solver.history["E_field"])

    animate_1d_phase_space(
        snapshots,
        time_hist,
        field_hist,
        output_path="docs/animations/two_stream_1d.gif",
        boxsize=boxsize,
        v_lim=6.5,
        fps=15,
        dpi=110,
    )


def generate_landau_damping_movie():
    print("\n[2/5] Generating 1D Landau Damping Movie...")
    Nx = 128
    k = 0.5
    boxsize = 2.0 * np.pi / k
    N_particles = 80000
    dt = 0.05
    t_end = 25.0
    v_th = 1.0
    alpha = 0.15  # Slightly larger for clear visual phase-mixing in movie

    weight = (1.0 * boxsize) / N_particles
    solver = PIC1DSolver(Nx=Nx, boxsize=boxsize, dt=dt)

    np.random.seed(42)
    u = np.linspace(0.0, 1.0, N_particles, endpoint=False)
    x = u * boxsize
    for _ in range(5):
        f = (x + (alpha / k) * np.sin(k * x)) / boxsize - u
        df = (1.0 + alpha * np.cos(k * x)) / boxsize
        x -= f / df
    x = np.mod(x, boxsize)

    vel = np.zeros((N_particles, 3))
    vel[:, 0] = np.random.normal(0.0, v_th, N_particles)

    solver.add_species("electrons", q=-weight, m=weight, pos=x, vel=vel)
    solver.initialize()

    snapshots = []
    def save_snap(s):
        snapshots.append({
            "time": s.time,
            "x": s.species[0].pos[:, 0].copy(),
            "vx": s.species[0].vel[:, 0].copy(),
            "E": s.E.copy(),
        })

    save_snap(solver)
    solver.run(t_end=t_end, callback=save_snap, callback_interval=5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=110)
    sub = slice(None, None, max(1, N_particles // 12000))

    sc = ax1.scatter(snapshots[0]["x"][sub], snapshots[0]["vx"][sub], s=0.7, color="#0077bb", alpha=0.4)
    ax1.set_xlim(0, boxsize)
    ax1.set_ylim(-4.0, 4.0)
    ax1.set_xlabel(r"$x$ ($c/\omega_{pe}$)")
    ax1.set_ylabel(r"$v_x$ / $v_{th}$")
    t1 = ax1.set_title(r"Phase-Mixing in Phase Space: $\omega_{pe} t = 0.0$", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    grid_x = solver.grid_x
    line_e, = ax2.plot(grid_x, snapshots[0]["E"], color="#d62728", lw=2)
    ax2.set_xlim(0, boxsize)
    ax2.set_ylim(-0.3, 0.3)
    ax2.set_xlabel(r"$x$ ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"Electric Field $E(x)$")
    t2 = ax2.set_title(r"Damping Electric Wave Profile", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    def update(frame):
        snap = snapshots[frame]
        sc.set_offsets(np.column_stack([snap["x"][sub], snap["vx"][sub]]))
        line_e.set_ydata(snap["E"])
        t1.set_text(rf"Phase-Mixing in Phase Space: $\omega_{{pe}} t = {snap['time']:.1f}$")
        t2.set_text(rf"Electric Field Damping: $\omega_{{pe}} t = {snap['time']:.1f}$")
        return sc, line_e, t1, t2

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=60)
    anim.save("docs/animations/landau_damping_1d.gif", writer=animation.PillowWriter(fps=15))
    plt.close(fig)
    print("  [Saved] docs/animations/landau_damping_1d.gif")


def generate_filamentation_2d_movie():
    print("\n[3/5] Generating 2D Filamentation Instability Movie...")
    Nx = 100
    Ny = 100
    Lx = 25.0
    Ly = 25.0
    N_particles = 60000
    dt = 0.08
    t_end = 22.0

    weight = (1.0 * Lx * Ly) / N_particles
    solver = PIC2DSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dt=dt)

    Nh = N_particles // 2
    np.random.seed(42)
    x1 = np.random.uniform(0.0, Lx, Nh)
    y1 = np.random.uniform(0.0, Ly, Nh)
    vel1 = np.zeros((Nh, 3))
    vel1[:, 0] = np.random.normal(2.5, 0.4, Nh)

    x2 = np.random.uniform(0.0, Lx, Nh)
    y2 = np.random.uniform(0.0, Ly, Nh)
    vel2 = np.zeros((Nh, 3))
    vel2[:, 0] = np.random.normal(-2.5, 0.4, Nh)

    solver.add_species("beam1", q=-weight, m=weight, pos_x=x1, pos_y=y1, vel=vel1)
    solver.add_species("beam2", q=-weight, m=weight, pos_x=x2, pos_y=y2, vel=vel2)
    solver.initialize()

    frames = [solver.rho.copy()]
    times = [0.0]

    def record_frame(s):
        frames.append(s.rho.copy())
        times.append(s.time)

    solver.run(t_end=t_end, callback=record_frame, callback_interval=5)

    animate_2d_density(
        frames,
        times,
        output_path="docs/animations/filamentation_2d.gif",
        Lx=Lx,
        Ly=Ly,
        fps=15,
        dpi=110,
    )


def generate_reconnection_2d_movie():
    print("\n[4/5] Generating 2D Magnetic Reconnection Movie...")
    Nx = 100
    Ny = 100
    Lx = 24.0
    Ly = 12.0
    N_particles = 60000
    dt = 0.05
    t_end = 16.0

    B0 = 1.0
    L_p = 1.5
    y_center = Ly / 2.0
    v_th = 0.4
    v_drift_z = 0.9

    weight = (1.0 * Lx * Ly) / N_particles
    q_macro = -weight
    m_macro = weight

    np.random.seed(42)
    y_parts = []
    while len(y_parts) < N_particles:
        y_c = np.random.uniform(0.0, Ly, N_particles)
        prob = (1.0 / np.cosh((y_c - y_center) / L_p)) ** 2 + 0.2
        p_r = np.random.uniform(0.0, 1.2, N_particles)
        y_parts.extend(y_c[p_r < prob].tolist())

    y_pos = np.array(y_parts[:N_particles])
    x_pos = np.random.uniform(0.0, Lx, N_particles)
    vel = np.zeros((N_particles, 3))
    vel[:, 0] = np.random.normal(0.0, v_th, N_particles)
    vel[:, 1] = np.random.normal(0.0, v_th, N_particles)
    vel[:, 2] = np.random.normal(v_drift_z, v_th, N_particles)

    delta_B = 0.12 * B0

    def compute_b_field(xp, yp):
        bx = B0 * np.tanh((yp - y_center) / L_p) - delta_B * np.cos(
            2.0 * np.pi * xp / Lx
        ) * np.sin(np.pi * (yp - y_center) / Ly)
        by = delta_B * (Ly / Lx) * np.sin(
            2.0 * np.pi * xp / Lx
        ) * np.cos(np.pi * (yp - y_center) / Ly)
        return np.column_stack((bx, by, np.zeros_like(bx)))

    rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
    _, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
    ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
    E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
    B_vec = compute_b_field(x_pos, y_pos)
    v_half = retard_velocity(vel, E_vec, B_vec, q_macro, m_macro, dt)

    gx = np.linspace(0.0, Lx, Nx)
    gy = np.linspace(0.0, Ly, Ny)
    GX, GY = np.meshgrid(gx, gy)
    GB = compute_b_field(GX.ravel(), GY.ravel())
    GBx = GB[:, 0].reshape((Ny, Nx))
    GBy = GB[:, 1].reshape((Ny, Nx))

    frames_rho = [rho.copy()]
    times = [0.0]
    Nt = int(t_end / dt)

    for step in range(Nt):
        ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
        E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
        B_vec = compute_b_field(x_pos, y_pos)
        v_next = boris_push(v_half, E_vec, B_vec, q_macro, m_macro, dt)

        x_pos += v_next[:, 0] * dt
        y_pos += v_next[:, 1] * dt
        x_pos = np.mod(x_pos, Lx)
        y_pos = np.clip(y_pos, 0.01, Ly - 0.01)

        rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
        _, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
        v_half = v_next

        if step % 6 == 0:
            frames_rho.append(rho.copy())
            times.append(step * dt)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    im = ax.imshow(frames_rho[0], origin="lower", extent=[0, Lx, 0, Ly], cmap="inferno", aspect="auto")
    st = ax.streamplot(gx, gy, GBx, GBy, color="cyan", linewidth=0.8, density=1.0)
    plt.colorbar(im, ax=ax, label=r"Plasma Density $\rho(x, y)$")
    ax.set_xlabel(r"x ($d_i$)")
    ax.set_ylabel(r"y ($d_i$)")
    title = ax.set_title(r"Magnetic Reconnection Sheet: $\omega_{pe} t = 0.0$", fontweight="bold")
    plt.tight_layout()

    def update(idx):
        im.set_data(frames_rho[idx])
        title.set_text(rf"Magnetic Reconnection Sheet: $\omega_{{pe}} t = {times[idx]:.1f}$")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames_rho), interval=80)
    anim.save("docs/animations/reconnection_2d.gif", writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print("  [Saved] docs/animations/reconnection_2d.gif")


def generate_expansion_3d_movie():
    print("\n[5/5] Generating 3D Plasma Expansion Movie...")
    Nx, Ny, Nz = 24, 24, 24
    Lx, Ly, Lz = 16.0, 16.0, 16.0
    N_particles = 25000
    dt = 0.06
    t_end = 6.0

    weight = (1.0 * Lx * Ly * Lz) / N_particles
    solver = PIC3DSolver(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dt=dt)

    np.random.seed(42)
    center = np.array([Lx / 2.0, Ly / 2.0, Lz / 2.0])
    pos_x = np.random.normal(center[0], 2.0, N_particles)
    pos_y = np.random.normal(center[1], 2.0, N_particles)
    pos_z = np.random.normal(center[2], 2.0, N_particles)
    pos_x = np.mod(pos_x, Lx)
    pos_y = np.mod(pos_y, Ly)
    pos_z = np.mod(pos_z, Lz)

    vel = np.zeros((N_particles, 3))
    solver.add_species("plasma", q=-weight, m=weight, pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, vel=vel)
    solver.initialize()

    snapshots = []
    sub = slice(None, None, 15)
    def save_snap(s):
        sp = s.species[0]
        snapshots.append((
            s.time,
            sp.x[sub].copy(),
            sp.y[sub].copy(),
            sp.z[sub].copy(),
            np.linalg.norm(sp.vel[sub], axis=1).copy(),
        ))

    save_snap(solver)
    solver.run(t_end=t_end, callback=save_snap, callback_interval=4)

    fig = plt.figure(figsize=(8, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    t0, x0, y0, z0, v0 = snapshots[0]
    sc = ax.scatter(x0, y0, z0, c=v0, cmap="plasma", s=2.0, alpha=0.6)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title(r"3D Spherical Coulomb Explosion: $\omega_{pe} t = 0.0$", fontweight="bold")
    plt.tight_layout()

    def update(frame_idx):
        t, xp, yp, zp, vp = snapshots[frame_idx]
        sc._offsets3d = (xp, yp, zp)
        sc.set_array(vp)
        # Azimuthal rotation for cinematic 3D perspective
        ax.view_init(elev=20, azim=frame_idx * 4.0)
        title.set_text(rf"3D Spherical Coulomb Explosion: $\omega_{{pe}} t = {t:.1f}$")
        return sc, title

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=80)
    anim.save("docs/animations/plasma_expansion_3d.gif", writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print("  [Saved] docs/animations/plasma_expansion_3d.gif")


def generate_asymmetric_reconnection_movie():
    print("\n[6/7] Generating Asymmetric Reconnection 2D Movie...")
    apply_epic_style()

    Nx = 100
    Ny = 100
    Lx = 24.0
    Ly = 12.0
    N_particles = 60000
    dt = 0.05
    t_end = 18.0

    B1 = 1.0
    B2 = 0.5
    n1 = 0.6
    n2 = 2.0
    L_p = 1.5
    y_center = Ly / 2.0
    v_th = 0.4
    v_drift_z = 0.8
    delta_B = 0.12 * min(B1, B2)

    avg_dens = 0.5 * (n1 + n2) + 0.5
    weight = (avg_dens * Lx * Ly) / N_particles
    q_macro = -weight
    m_macro = weight

    np.random.seed(42)
    y_parts = []
    max_dens = 1.2 * max(n1, n2) + 1.0
    while len(y_parts) < N_particles:
        y_c = np.random.uniform(0.0, Ly, N_particles)
        th = np.tanh((y_c - y_center) / L_p)
        sech2 = (1.0 / np.cosh((y_c - y_center) / L_p)) ** 2
        prob = 0.8 * sech2 + n1 * 0.5 * (1.0 + th) + n2 * 0.5 * (1.0 - th)
        p_r = np.random.uniform(0.0, max_dens, N_particles)
        y_parts.extend(y_c[p_r < prob].tolist())

    y_pos = np.array(y_parts[:N_particles])
    x_pos = np.random.uniform(0.0, Lx, N_particles)
    vel = np.zeros((N_particles, 3))
    vel[:, 0] = np.random.normal(0.0, v_th, N_particles)
    vel[:, 1] = np.random.normal(0.0, v_th, N_particles)
    sheet_env = 1.0 / np.cosh((y_pos - y_center) / L_p)
    vel[:, 2] = np.random.normal(v_drift_z * sheet_env, v_th, N_particles)

    def compute_b(xp, yp):
        th = np.tanh((yp - y_center) / L_p)
        bx_asym = 0.5 * (B1 + B2) * th + 0.5 * (B1 - B2)
        env = np.exp(-((yp - y_center) / (2.0 * L_p)) ** 2)
        bx_pert = -delta_B * np.cos(2.0 * np.pi * xp / Lx) * np.sin(np.pi * (yp - y_center) / Ly) * env
        by_pert = delta_B * (Ly / Lx) * np.sin(2.0 * np.pi * xp / Lx) * np.cos(np.pi * (yp - y_center) / Ly) * env
        return np.column_stack((bx_asym + bx_pert, by_pert, np.zeros_like(bx_asym)))

    rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
    _, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
    ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
    E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
    B_vec = compute_b(x_pos, y_pos)
    v_half = retard_velocity(vel, E_vec, B_vec, q_macro, m_macro, dt)

    gx = np.linspace(0.0, Lx, Nx)
    gy = np.linspace(0.0, Ly, Ny)
    GX, GY = np.meshgrid(gx, gy)
    GB = compute_b(GX.ravel(), GY.ravel())
    GBx = GB[:, 0].reshape((Ny, Nx))
    GBy = GB[:, 1].reshape((Ny, Nx))

    frames = [rho.copy()]
    times = [0.0]
    Nt = int(t_end / dt)

    for step in range(Nt):
        ex_p, ey_p = interpolate_field_2d(x_pos, y_pos, Ex, Ey, Nx, Ny, Lx, Ly)
        E_vec = np.column_stack((ex_p, ey_p, np.zeros_like(ex_p)))
        B_vec = compute_b(x_pos, y_pos)
        v_next = boris_push(v_half, E_vec, B_vec, q_macro, m_macro, dt)

        x_pos += v_next[:, 0] * dt
        y_pos += v_next[:, 1] * dt
        x_pos = np.mod(x_pos, Lx)
        y_pos = np.clip(y_pos, 0.05, Ly - 0.05)

        rho = deposit_charge_2d(x_pos, y_pos, q_macro, Nx, Ny, Lx, Ly)
        _, Ex, Ey = solve_poisson_2d_fft(rho, Lx, Ly)
        v_half = v_next

        if step % 6 == 0:
            frames.append(rho.copy())
            times.append(step * dt)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=110, facecolor=EPIC_COLORS["bg_dark"])
    ax.set_facecolor(EPIC_COLORS["bg_axes"])
    im = ax.imshow(frames[0], origin="lower", extent=[0, Lx, 0, Ly], cmap="inferno", aspect="auto")
    ax.streamplot(gx, gy, GBx, GBy, color="#ffffff", linewidth=0.7, density=1.0)
    cbar = plt.colorbar(im, ax=ax, label=r"Plasma Density $\rho(x, y)$")
    cbar.ax.yaxis.label.set_color(EPIC_COLORS["text"])
    ax.set_xlabel(r"x ($c/\omega_{pe}$)")
    ax.set_ylabel(r"y ($c/\omega_{pe}$)")
    title = ax.set_title(r"Asymmetric Dayside Reconnection: $\omega_{pe} t = 0.0$", fontweight="bold")
    plt.tight_layout()

    def update(idx):
        im.set_data(frames[idx])
        title.set_text(rf"Asymmetric Dayside Reconnection: $\omega_{{pe}} t = {times[idx]:.1f}$")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=80)
    os.makedirs("docs/animations", exist_ok=True)
    anim.save("docs/animations/asymmetric_reconnection_2d.gif", writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print("  [Saved] docs/animations/asymmetric_reconnection_2d.gif")


def generate_electrostatic_shock_movie():
    print("\n[7/7] Generating Electrostatic Shock 1D Movie...")
    apply_epic_style()

    Nx = 300
    boxsize = 50.0
    dx = boxsize / Nx
    dt = 0.05
    t_end = 25.0
    Nt = int(t_end / dt)

    N_electrons = 80000
    N_ions = 80000
    m_i_ratio = 16.0
    T_e = 1.0
    T_i = 0.05
    v_th_e = np.sqrt(T_e)
    v_th_i = np.sqrt(T_i / m_i_ratio)
    c_s = np.sqrt(T_e / m_i_ratio)
    v_drift = 2.0 * c_s

    weight_e = (1.0 * boxsize) / N_electrons
    q_e = -1.0 * weight_e
    m_e = 1.0 * weight_e
    weight_i = (1.0 * boxsize) / N_ions
    q_i = 1.0 * weight_i
    m_i = m_i_ratio * weight_i

    np.random.seed(42)
    x_e = np.random.uniform(0.0, boxsize, N_electrons)
    x_i = np.random.uniform(0.0, boxsize, N_ions)
    v_e = np.random.normal(0.0, v_th_e, (N_electrons, 3))
    v_i = np.random.normal(0.0, v_th_i, (N_ions, 3))

    left_e = x_e < (boxsize / 2.0)
    v_e[left_e, 0] += v_drift
    v_e[~left_e, 0] -= v_drift
    left_i = x_i < (boxsize / 2.0)
    v_i[left_i, 0] += v_drift
    v_i[~left_i, 0] -= v_drift

    rho = deposit_charge_1d(x_e, q_e, Nx, boxsize) + deposit_charge_1d(x_i, q_i, Nx, boxsize)
    phi, E = solve_poisson_1d_fft(rho, boxsize)

    E_e = interpolate_field_1d(x_e, E, Nx, boxsize).ravel()
    E_i = interpolate_field_1d(x_i, E, Nx, boxsize).ravel()
    E_e_vec = np.column_stack((E_e, np.zeros((N_electrons, 2))))
    E_i_vec = np.column_stack((E_i, np.zeros((N_ions, 2))))
    B_vec_e = np.zeros((N_electrons, 3))
    B_vec_i = np.zeros((N_ions, 3))

    v_e_half = retard_velocity(v_e, E_e_vec, B_vec_e, q_e, m_e, dt)
    v_i_half = retard_velocity(v_i, E_i_vec, B_vec_i, q_i, m_i, dt)

    snaps = []
    sub = slice(None, None, max(1, N_ions // 10000))

    for step in range(Nt):
        E_e = interpolate_field_1d(x_e, E, Nx, boxsize).ravel()
        E_i = interpolate_field_1d(x_i, E, Nx, boxsize).ravel()
        E_e_vec[:, 0] = E_e
        E_i_vec[:, 0] = E_i

        v_e_next = boris_push(v_e_half, E_e_vec, B_vec_e, q_e, m_e, dt)
        v_i_next = boris_push(v_i_half, E_i_vec, B_vec_i, q_i, m_i, dt)

        x_e = np.mod(x_e + v_e_next[:, 0] * dt, boxsize)
        x_i = np.mod(x_i + v_i_next[:, 0] * dt, boxsize)

        rho = deposit_charge_1d(x_e, q_e, Nx, boxsize) + deposit_charge_1d(x_i, q_i, Nx, boxsize)
        phi, E = solve_poisson_1d_fft(rho, boxsize)

        v_e_half = v_e_next
        v_i_half = v_i_next

        if step % 7 == 0:
            snaps.append((step * dt, x_i[sub].copy(), (v_i_half[sub, 0] / c_s).copy(), phi.copy()))

    grid_x = np.linspace(0.0, boxsize, Nx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=110, facecolor=EPIC_COLORS["bg_dark"])
    ax1.set_facecolor(EPIC_COLORS["bg_axes"])
    ax2.set_facecolor(EPIC_COLORS["bg_axes"])

    t0, x0, v0, p0 = snaps[0]
    sc = ax1.scatter(x0, v0, s=0.9, color=EPIC_COLORS["cyan"], alpha=0.6)
    ax1.set_xlim(0, boxsize)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax1.set_ylabel(r"Ion Velocity $v_{i, x} / c_s$")
    t1 = ax1.set_title(r"Ion Phase Space: $\omega_{pe} t = 0.0$", fontweight="bold")

    line_phi, = ax2.plot(grid_x, p0, color=EPIC_COLORS["emerald"], lw=2.2)
    ax2.set_xlim(0, boxsize)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_xlabel(r"Position $x$ ($c/\omega_{pe}$)")
    ax2.set_ylabel(r"Electrostatic Potential $\phi$")
    t2 = ax2.set_title(r"Electrostatic Shock Barrier $\Delta \phi$", fontweight="bold")
    plt.tight_layout()

    def update(frame):
        t_now, xp, vp, phip = snaps[frame]
        sc.set_offsets(np.column_stack([xp, vp]))
        line_phi.set_ydata(phip)
        t1.set_text(rf"Ion Shock Phase Space: $\omega_{{pe}} t = {t_now:.1f}$")
        t2.set_text(rf"Shock Potential Barrier: $\omega_{{pe}} t = {t_now:.1f}$")
        return sc, line_phi, t1, t2

    anim = animation.FuncAnimation(fig, update, frames=len(snaps), interval=70)
    anim.save("docs/animations/electrostatic_shock_1d.gif", writer=animation.PillowWriter(fps=14))
    plt.close(fig)
    print("  [Saved] docs/animations/electrostatic_shock_1d.gif")


if __name__ == "__main__":
    os.makedirs("docs/animations", exist_ok=True)
    generate_two_stream_movie()
    generate_landau_damping_movie()
    generate_filamentation_2d_movie()
    generate_reconnection_2d_movie()
    generate_expansion_3d_movie()
    generate_asymmetric_reconnection_movie()
    generate_electrostatic_shock_movie()
    print("\nAll 7 scientific simulation movies generated successfully!")
