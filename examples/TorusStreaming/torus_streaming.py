import numpy as np
import matplotlib.pyplot as plt
import os
from pyaxisymflow.utils.plotset import plotset
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.utils.dump_vtk import vtk_init, vtk_write
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_unb
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.FDM_stokes_phi_solve import (
    stokes_phi_init,
    stokes_phi_solve_LU,
)
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3
from pyaxisymflow.kernels.compute_velocity_from_phi import compute_velocity_from_phi_unb

# global settings
grid_size_z = 256
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
eps = np.finfo(float).eps
num_threads = 4

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e4
moll_zone = dx * 2**0.5
r_core = 0.125
a_by_r_core = 0.34146341463414637
a = a_by_r_core * r_core
AR = 0.4285714285714286
b = a * AR
freq = 16
freqTimer_limit = 1 / freq
no_cycles = 20
snaps_per_cycle = 40
fotoTimer_limit = 1 / freq / snaps_per_cycle
snap_cycle_start = 19
snap_cycle_end = 20

omega = 2 * np.pi * freq
d_AC_by_r_core = 0.11235582096628798
e = 0.0675
U_0 = e * a * omega
nu = omega * (d_AC_by_r_core * r_core) ** 2
tEnd = no_cycles / freq

# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)


# load initial conditions
vorticity = 0 * Z
penal_vorticity = 0 * Z
temp_vorticity = 0 * Z
psi = 0 * Z
u_z = 0 * Z
u_r = 0 * Z
u_z_pen = 0 * Z
u_r_pen = 0 * Z
avg_psi = 0 * Z
avg_u_z = 0 * Z
avg_u_r = 0 * Z
Z_cm = 0.5
R_cm = r_core
t = 0
fotoTimer = 0.0
it = 0
freqTimer = 0.0

vel_phi = 0 * Z
u_z_divg = 0 * Z
u_r_divg = 0 * Z
vel_divg = 0 * Z
u_z_breath = 0 * Z
u_r_breath = 0 * Z
u_breath_divg = 0 * Z

#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2 / AR**2) + a
char_func = 0 * Z
char_func0 = 0 * Z
smooth_Heaviside(char_func0, phi0, moll_zone)
smooth_Heaviside(char_func, phi0, moll_zone)
d = np.ma.array(char_func, mask=char_func < 0.5)

FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
vtk_image_data, temp_vtk_array, writer = vtk_init(grid_size_z, grid_size_r)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)
_, _, LU_decomp_phi = stokes_phi_init(grid_size_z, grid_size_r, dx, R)

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
        plt.contourf(Z, R, avg_psi, levels=25, extend="both", cmap=lab_cmp)
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
        plt.contourf(Z, R, d, cmap="Greys", zorder=2)
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()

        avg_psi[...] *= 0
        avg_u_z[...] *= 0
        avg_u_r[...] *= 0

    # move body and get char func
    R_cm_t = R_cm + e * a * np.sin(omega * t)
    phi = a - np.sqrt((Z - Z_cm) ** 2 + (R - R_cm_t) ** 2 / AR**2)
    char_func *= 0
    smooth_Heaviside(char_func, phi, moll_zone)

    # solve for potential function and get velocity
    vel_divg[...] = U_0 * np.cos(omega * t) / R
    stokes_phi_solve_LU(
        vel_phi, LU_decomp_phi, char_func * vel_divg, grid_size_z, grid_size_r
    )
    compute_velocity_from_phi_unb(u_z_divg, u_r_divg, vel_phi, dx)
    u_z[...] += u_z_divg
    u_r[...] += u_r_divg

    # get dt
    dt = min(
        0.9 * dx**2 / 4 / nu,
        CFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if t + dt > tEnd:
        dt = tEnd - t

    # integrate averaged fields
    avg_psi[...] += psi * dt
    avg_u_z[...] += u_z * dt
    avg_u_r[...] += u_r * dt

    # penalise velocity
    brinkmann_penalize(
        brink_lam,
        dt,
        char_func,
        u_z_breath,
        U_0 * np.cos(omega * t) + u_r_breath,
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

    #  update time
    t += dt
    it += 1
    freqTimer = freqTimer + dt
    if t > fotoTimer_start and t < fotoTimer_end:
        fotoTimer += dt
    if it % 100 == 0:
        print(t, np.amax(vorticity))


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
os.system("rm -f *png")
