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

# from kernels.poisson_solve_unb import pseudo_poisson_solve_unb

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
fotoTimer_limit = 0.1
brink_lam = 1e12
moll_zone = dx * 2 ** 0.5
r_cyl = 0.075
U_0 = 1.0
Re = 100.0
nu = U_0 * 2 * r_cyl / Re
nondim_T = 300
tEnd = nondim_T * r_cyl / U_0
T_ramp = 20 * r_cyl / U_0

# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

# load initial conditions
vorticity = 0 * Z
penal_vorticity = 0 * Z
temp_vorticity = 0 * Z
psi = 0 * Z
avg_psi = 0 * Z
avg_vort = 0 * Z
avg_part_char_func = 0 * Z
u_z = 0 * Z
u_r = 0 * Z
u_z_old =  0 * Z
u_r_old =  0 * Z
u_z_upen = 0 * Z
u_r_upen = 0 * Z
inside_bubble = 0 * Z
u_z_breath = 0 * Z
u_r_breath = 0 * Z
Z_double, R_double = np.meshgrid(z, z)
z_particles = Z_double.copy()
r_particles = R_double.copy()
vort_double = 0 * Z_double
vort_particles = 0 * vort_double
Z_double, R_double = np.meshgrid(z, z)
z_particles = Z_double.copy()
r_particles = R_double.copy()
vort_double = 0 * Z_double
vort_particles = 0 * vort_double

fotoTimer = 0.0
it = 0
freqTimer = 0.0
freq = 8.0
freqTimer_limit = 1 / freq
Z_cm = 0.25
R_cm = 0.0
t = 0.0
T = []
diff = 0
F_total = 0
#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)
part_mass = np.sum(char_func * R)

FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
vtk_image_data, temp_vtk_array, writer = vtk_init()

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)
   

    # solve for stream function and get velocity
    FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)
    
    # add free stream
    prefac_x = 1.0
    prefac_y = 0.0
    if t < T_ramp:
        prefac_x = np.sin(0.5 * np.pi * t / T_ramp)
        prefac_y = 5e-2 * np.sin(np.pi * t / T_ramp)
    u_z[...] += U_0 * prefac_x
    u_r[...] += U_0 * prefac_y


    if fotoTimer >= fotoTimer_limit or t == 0:
        fotoTimer = 0.0

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
            u_z,
            levels=[
                0.0,
            ],
            colors="red",
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
    dt = min(
        0.9 * dx ** 2 / 4 / nu,
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )

    
    # integrate averaged fields
    avg_psi[...] += psi * dt


    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, char_func, 0.0, 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    )
    vorticity[...] += penal_vorticity
    Cd =  2*2 * np.pi * dx * dx * brink_lam * np.sum(R*char_func * (u_z))/ (np.pi*r_cyl**2)
    



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
        print(t, np.amax(vorticity), Cd)


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
