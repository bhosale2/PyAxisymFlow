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
from pyaxisymflow.kernels.advect_particle import advect_vorticity_via_particles
from pyaxisymflow.kernels.compute_forces import compute_force_on_body
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_unb
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3


# global settings
grid_size_z = 512
domain_AR = 0.3
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
eps = np.finfo(float).eps
num_threads = 4

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
fotoTimer_limit = 0.1
brink_lam = 1e4
moll_zone = dx * 2**0.5
r_cyl = 0.025
U_0 = 1.0
nondim_T = 1000
tEnd = 10
T_ramp = 20 * r_cyl / U_0
rho_f = 1


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
u_z_upen = 0 * Z
u_r_upen = 0 * Z

fotoTimer = 0.0
it = 0
Z_cm = 0.60
R_cm = 0.0
t = 0.0
T = []
rho_s = 1.1
#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)
part_mass = 2 * np.pi * dx * dx * rho_s * np.sum(char_func * R)
part_vol = 2 * np.pi * dx * dx * np.sum(char_func * R)
t_vel = part_vol * g / (6 * np.pi * nu * r_cyl)
Re = 0.1
nu = rho_f * t_vel * 2 * r_cyl / Re

part_Z_cm_old = Z_cm
part_Z_cm_new = Z_cm
part_Z_cm = Z_cm

FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)

F_un = 0
F_pen = 0.0
U_z_cm_part = 0
U_z_cm_part_old = 0
diff = 0
T = []
U_z_cm = []
CD = []

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    if fotoTimer >= fotoTimer_limit or t == 0:
        fotoTimer = 0.0
        levels = np.linspace(-0.1, 0.1, 25)
        plt.contourf(
            Z,
            R,
            vorticity,
            levels=100,
            extend="both",
            cmap=lab_cmp,
        )

        plt.contour(
            Z,
            R,
            char_func,
            levels=[
                0.5,
            ],
            colors="grey",
        )

        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()
        plt.close("all")

    # get dt
    dt = 1 * min(
        0.9 * dx**2 / 4 / nu,
        CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
    )

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, char_func, U_z_cm_part, 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    )
    vorticity[...] += penal_vorticity

    F_pen, F_un = compute_force_on_body(
        R, char_func, rho_f, brink_lam, u_z, U_z_cm_part, part_vol, dt, diff
    )
    F_total = 2 * np.pi * dx * dx * F_pen + F_un
    phi0 = -np.sqrt((Z - part_Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
    char_func = 0 * Z
    smooth_Heaviside(char_func, phi0, moll_zone)
    advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)
    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    Cd = (
        2
        * 2
        * np.pi
        * dx
        * dx
        * brink_lam
        * np.sum(R * char_func * (u_z))
        / (np.pi * r_cyl**2)
    )
    # (Cd = F/0.5*p*U^2*A)
    # update particle location and velocity
    g = (rho_s - rho_f) * 9.81
    U_z_cm_part_old = U_z_cm_part
    U_z_cm_part += dt * ((-g + (F_total / part_mass)))
    diff = dt * (-g / rho_s + (F_total / part_mass))
    part_Z_cm_new = part_Z_cm
    part_Z_cm += U_z_cm_part_old * dt + (
        0.5 * dt * dt * (-g / rho_s + (F_total / part_mass))
    )
    part_Z_cm_old = part_Z_cm_new
    T = np.append(T, t)
    U_z_cm = np.append(U_z_cm, U_z_cm_part)
    CD = np.append(CD, Cd)

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    if it % 50 == 0:
        print(f"time: {t}, max vort: {np.amax(vorticity)},drag_coeff:{Cd}")


np.savetxt(
    "velocity_drag.csv",
    np.c_[
        np.array(T) * omega,
        np.array(U_z_cm),
        np.array(CD),
    ],
    delimiter=",",
)
os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
    "-vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
