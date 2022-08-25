import os

import matplotlib.pyplot as plt
import numpy as np
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_velocity_from_phi import compute_velocity_from_phi_unb
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_unb
from pyaxisymflow.kernels.FastDiagonalisationPotentialSolver import (
    FastDiagonalisationPotentialSolver,
)
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.utils.plotset import plotset

# global settings
grid_size_z = 256
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
LCFL = 0.1
eps = np.finfo(float).eps
num_threads = 4

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e4
moll_zone = dx * 2**0.5

# torus geometry details
r_core = 0.096  # core radius
a_by_r_core = 0.4  # tube-to-core radius ratio
a0 = a_by_r_core * r_core  # tube radius

# particle geometry details
beta = 0.1
part_rad = beta * r_core

# flow and actuation parameter
Re_t = 0.1  # translational reynolds
zeta = 6.25  # oscillatory-translational vel ratio
U_mean = 0.05
nu = U_mean * a0 / Re_t
U_osc = zeta * U_mean
e = 0.1
e_r_core = e * a_by_r_core
omega = U_osc / a0 / e
freq = omega / 2 / np.pi

# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

# load initial conditions
# initialize fields
vorticity = 0 * Z
penal_vorticity = 0 * Z
temp_vorticity = 0 * Z
psi = 0 * Z
phi = 0 * Z
u_z = 0 * Z
u_r = 0 * Z
u_z_pen = 0 * Z
u_r_pen = 0 * Z
vel_phi = 0 * Z
u_z_divg = 0 * Z
u_r_divg = 0 * Z
vel_divg = 0 * Z
u_z_breath = 0 * Z
u_r_breath = 0 * Z
u_breath_divg = 0 * Z
avg_psi = 0 * Z
avg_u_z = 0 * Z
avg_u_r = 0 * Z
# torus initial config
Z_cm = 0.3
Z_cm_t = Z_cm
R_cm = r_core
Z_cm_end = 0.7
a = a0
# particle initial config
part_Z_cm_offset = 0.0 * a
part_Z_cm = Z_cm + part_Z_cm_offset
part_R_cm = 0.0
part_loc = []
# initialize time-related variables
t = 0
it = 0
freqTimer = 0.0
dataTimer = 0.0
restartSaveTimer = 5000
tEnd = (Z_cm_end - Z_cm) / U_mean
if zeta == 0:
    freqTimer_limit = 0.005 * tEnd
    dataTimer_limit = tEnd / 1000.0
else:
    freqTimer_limit = 1 / freq
    data_per_cycle = 100
    dataTimer_limit = 1 / freq / data_per_cycle

# Account for restarts
if os.path.exists("restart.npz"):
    restart = np.load("restart.npz")
    t = float(restart["t"])
    vorticity = restart["vorticity"]
    part_Z_cm = float(restart["part_Z_cm"])
    part_R_cm = float(restart["part_R_cm"])
    part_loc = restart["part_loc"].tolist()
    print(f"Restarting with")
    print(f"t = {t}")
    print(f"part z cm : {part_Z_cm}")
    print(f"part r cm : {part_R_cm}")
else:
    print("starting from scratch...")
    os.system("rm -f *.mp4 *.png *.csv")

# create char function
torus_phi = a - np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2)
torus_char_func = 0 * Z
smooth_Heaviside(torus_char_func, torus_phi, moll_zone)

part_phi = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + part_rad
part_char_func = 0 * Z
smooth_Heaviside(part_char_func, part_phi, moll_zone)

# initialize kernels
FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
FD_potential_solver = FastDiagonalisationPotentialSolver(grid_size_r, grid_size_z, dx)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        plt.contourf(
            Z,
            R,
            avg_psi,
            levels=25,
            extend="both",
            cmap=lab_cmp,
        )
        plt.colorbar()
        plt.contour(
            Z,
            R,
            avg_psi,
            levels=[
                0.0,
            ],
            colors="grey",
        )
        plt.contour(Z, R, torus_char_func, levels=[0.5], colors="k")
        plt.contour(Z, R, part_char_func, levels=[0.5], colors="k")
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()

        avg_psi[...] = 0.0
        avg_u_z[...] = 0.0
        avg_u_r[...] = 0.0

    if it % restartSaveTimer == 0 and it != 0:
        # save for restart
        np.savez(
            "restart.npz",
            t=t,
            vorticity=vorticity,
            part_Z_cm=part_Z_cm,
            part_R_cm=part_R_cm,
            part_loc=np.array(part_loc),
        )
        np.savez(
            f"restart{it:06d}.npz",
            t=t,
            vorticity=vorticity,
            part_Z_cm=part_Z_cm,
            part_R_cm=part_R_cm,
            part_loc=np.array(part_loc),
        )
        # save particle trajectory
        np.savetxt("part_loc.csv", np.array(part_loc), delimiter=",")

    # move body and get char func
    R_cm_t = R_cm + e * a0 * np.sin(omega * t)
    Z_cm_t = Z_cm + U_mean * t
    # varying cross section to conserve volume
    a = a0 / np.sqrt(1 + e_r_core * np.sin(omega * t))
    phi = a - np.sqrt((Z - Z_cm_t) ** 2 + (R - R_cm_t) ** 2)
    torus_char_func *= 0
    smooth_Heaviside(torus_char_func, phi, moll_zone)

    # compute internal velocities in torus
    da_dt = (
        -0.5
        * a0
        * e_r_core
        * omega
        * np.cos(omega * t)
        * (1 + e_r_core * np.sin(omega * t)) ** -1.5
    )
    u_z_breath[...] = da_dt * (Z - Z_cm_t) / a
    u_r_breath[...] = da_dt * (R - R_cm_t) / a
    u_breath_divg[1:-1, 1:-1] = (u_r_breath[2:, 1:-1] - u_r_breath[:-2, 1:-1]) / (
        2 * dx
    )
    +(u_r_breath[1:-1, 1:-1] / R[1:-1, 1:-1]) + (
        u_z_breath[1:-1, 2:] - u_z_breath[1:-1, :-2]
    ) / (2 * dx)

    # solve for potential function and get velocity
    vel_divg[...] = U_osc * np.cos(omega * t) / R
    vel_divg[...] += u_breath_divg
    FD_potential_solver.solve(
        solution_field=vel_phi, rhs_field=(torus_char_func * vel_divg)
    )
    compute_velocity_from_phi_unb(u_z_divg, u_r_divg, vel_phi, dx)
    u_z[...] += u_z_divg
    u_r[...] += u_r_divg

    # get dt
    dt = min(
        0.9 * dx**2 / 4 / nu,
        CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if dataTimer >= dataTimer_limit or t == 0 or t == tEnd:
        dataTimer = 0.0
        part_loc.append([t, (Z_cm_t - part_Z_cm) / a0])
    if t + dt > tEnd:
        dt = tEnd - t

    # integrate averaged fields
    avg_psi[...] += psi * dt
    avg_u_z[...] += u_z * dt
    avg_u_r[...] += u_r * dt

    # get particle phi
    part_phi[...] = -np.sqrt((Z - part_Z_cm) ** 2 + (R - part_R_cm) ** 2) + part_rad
    part_char_func[...] *= 0
    smooth_Heaviside(part_char_func, part_phi, moll_zone)

    # projection
    U_z = np.sum(part_char_func * u_z * R) / np.sum(part_char_func * R)

    # penalise torus velocity
    brinkmann_penalize(
        brink_lam,
        dt,
        torus_char_func,
        U_mean + u_z_breath,
        U_osc * np.cos(omega * t) + u_r_breath,
        u_z,
        u_r,
        u_z_pen,
        u_r_pen,
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z_pen - u_z, u_r_pen - u_r, dx
    )
    vorticity[...] += penal_vorticity

    # penalise particle velocity
    u_z[...] = u_z_pen
    u_r[...] = u_r_pen
    brinkmann_penalize(
        brink_lam,
        dt,
        part_char_func,
        U_z,
        0.0,
        u_z,
        u_r,
        u_z_pen,
        u_r_pen,
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z_pen - u_z, u_r_pen - u_r, dx
    )
    vorticity[...] += penal_vorticity

    # advect vorticity
    advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    # move particle
    part_Z_cm += dt * U_z

    # stop simulation if particle goes out of domain
    if part_Z_cm <= part_rad:
        part_loc.append(
            [-1.0, -1.0]
        )  # some bogus value to mark sim ended due to particle escaping domain
        print("Particle went out of domain.")
        break

    #  update time
    t += dt
    it += 1
    freqTimer = freqTimer + dt
    dataTimer = dataTimer + dt
    if it % 100 == 0:
        print(t, dt, np.amax(vorticity))

part_loc = np.array(part_loc)
np.savetxt("part_loc.csv", part_loc, delimiter=",")
plt.plot(part_loc[:, 0], part_loc[:, 1])
plt.savefig("traj.png")

os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
    "-vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
os.system("rm -f *png")
