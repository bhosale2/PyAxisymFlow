import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../../")
from set_sim_params import grid_size_z, grid_size_r, dx, eps, LCFL, domain_AR
from utils.plotset import plotset
from utils.custom_cmap import lab_cmp
from utils.dump_vtk import vtk_init, vtk_write
from kernels.brinkmann_penalize import brinkmann_penalize
from kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from kernels.compute_vorticity_from_velocity import compute_vorticity_from_velocity_unb
from kernels.advect_particle import advect_vorticity_via_particles
from kernels.compute_forces import compute_force_on_body
from kernels.smooth_Heaviside import smooth_Heaviside
from kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from kernels.diffusion_RK2_unb import diffusion_RK2_unb
from kernels.FastDiagonalisationStokesSolver import FastDiagonalisationStokesSolver

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
fotoTimer_limit = 0.1
brink_lam = 1e4
moll_zone = dx * 2 ** 0.5
r_cyl = 0.075
freq = 16
freqTimer_limit = 1 / freq
omega = 2 * np.pi * freq
M_sq = 100.2
nond_AC = 1.0 / np.sqrt(M_sq)
e = 0.1
U_0 = e * r_cyl * omega
Rs = (e / nond_AC) ** 2
nu = e * U_0 * r_cyl / Rs
no_cycles = 25
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
avg_psi = 0 * Z
Z_cm = 0.5
R_cm = 0.0
t = 0
fotoTimer = 0.0
it = 0
freqTimer = 0.0
u_z_upen = 0 * Z
u_r_upen = 0 * Z
Z_double, R_double = np.meshgrid(z, z)
vort_double = 0 * Z_double
vort_particles = 0 * vort_double
Z_double, R_double = np.meshgrid(z, z)
z_particles = Z_double.copy()
r_particles = R_double.copy()


#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)

FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
vtk_image_data, temp_vtk_array, writer = vtk_init()

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    # stokes_psi_solve_gmres(psi, FDM_matrix, precond_matrix, vorticity, R)
    FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        fotoTimer = 0.0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(Z, R, avg_psi, levels=25, extend="both", cmap=lab_cmp)
        plt.contour(
            Z,
            R,
            avg_psi,
            levels=[
                0.0,
            ],
            colors="grey",
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
        ax.set_aspect(aspect=1)
        plt.savefig(
            "snap_temp" + str("%0.4d" % (t * 100)) + ".png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )       
        vtk_write(
            "axisym_avg_" + str("%0.4d" % (t * 100)) + ".vti",
            vtk_image_data,
            temp_vtk_array,
            writer,
            ["char_func", "avg_psi"],
            [char_func, avg_psi],
        )
        avg_psi[...] = 0 * Z

    # get dt
    dt = min(
        0.9 * dx ** 2 / 4 / nu,
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if fotoTimer + dt > fotoTimer_limit:
        dt = fotoTimer_limit - fotoTimer
    if t + dt > tEnd:
        dt = tEnd - t

    # integrate averaged fields
    avg_psi[...] += psi * dt

    # move body
    Z_cm_t = Z_cm + e * r_cyl * np.sin(omega * t)
    phi = r_cyl - np.sqrt((Z - Z_cm_t) ** 2 + (R - R_cm) ** 2)
    char_func = 0 * Z
    smooth_Heaviside(char_func, phi, moll_zone)
    # penalise velocity
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, char_func, U_0 * np.cos(omega * t), 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(penal_vorticity, -u_z_upen +u_z, -u_r_upen + u_r, dx)
    vorticity[...] += penal_vorticity

    advect_vorticity_via_particles(
        z_particles, r_particles, vort_particles, vorticity, Z_double, R_double, grid_size_r, u_z, u_r, dx, dt
    )

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    freqTimer = freqTimer + dt
    if it % 50 == 0:
        print(t, np.amax(vorticity))


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
)