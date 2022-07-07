import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../../")
from sympy import exp, I, conjugate, diff, lambdify, symbols, re, im, besselj, bessely, simplify
from set_sim_params import grid_size_z, grid_size_r, dx, eps, LCFL, domain_AR
from utils.plotset import plotset
from utils.custom_cmap import lab_cmp
from utils.dump_vtk import vtk_init, vtk_write
from kernels.brinkmann_penalize import brinkmann_penalize
from kernels.compute_forces import compute_force_on_body
from kernels.smooth_Heaviside import smooth_Heaviside
from kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from kernels.FastDiagonalisationStokesSolver import FastDiagonalisationStokesSolver
from kernels.periodic_boundary_ghost_comm import gen_periodic_boundary_ghost_comm
from kernels.compute_velocity_from_psi import compute_velocity_from_psi_periodic
from kernels.compute_vorticity_from_velocity import compute_vorticity_from_velocity_periodic
from kernels.advect_particle import advect_vorticity_via_particles_periodic
from kernels.diffusion_RK2 import diffusion_RK2_periodic
from kernels.theory_rigid_slab import *
#plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e4
moll_zone = dx * 2 ** 0.5
wall_thickness = 0.05
R_wall_center = 0.3
R_tube = R_wall_center-wall_thickness
Re = 25
U_0 = 1.0
R_extent = 0.5
U_act = U_0 * (R_extent / R_tube) ** 2
nu = U_0 * 2 * R_tube / Re
nondim_T = 300
tEnd = nondim_T * R_tube  / U_0
T_ramp = 20 * R_tube  / U_0
freqTimer = 0.0
freq = 8.0
freqTimer_limit = 1 / freq
omega = 2*np.pi*freq
ghost_size = 2
per_communicator = gen_periodic_boundary_ghost_comm(ghost_size)
fotoTimer_limit = 0.1

y_range=  np.linspace(0,R_tube,100)
Soln = theory_axisymmetric_rigid_slab_spatial(omega,nu,R_tube)

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
fotoTimer = 0.0
it = 0
Z_cm = 0

t = 0.0
T = []
diff = 0
F_total = 0
#  create char function
phi0 = -np.sqrt((R - R_wall_center) ** 2) + wall_thickness
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)
part_mass = np.sum(char_func * R)

psi_inner = psi[..., ghost_size:-ghost_size].copy()
FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z-2*ghost_size, dx, bc_type= "homogenous_neumann_along_r_and_periodic_along_z")
vtk_image_data, temp_vtk_array, writer = vtk_init()

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    psi_inner[...] = psi[..., ghost_size:-ghost_size]
    FD_stokes_solver.solve(
        solution_field=psi_inner, rhs_field=vorticity[:, ghost_size:-ghost_size]
    )
    psi[..., ghost_size:-ghost_size] = psi_inner
    compute_velocity_from_psi_periodic(u_z, u_r, psi, R, dx, per_communicator)

    if fotoTimer >= fotoTimer_limit or t == 0:
        fotoTimer = 0.0

        fotoTimer = 0.0
        levels = np.linspace(-0.1, 0.1, 25)
        plt.plot(R[:,int(grid_size_z/2)],u_z[:,int(grid_size_z/2)])
        plt.plot(y_range, theory_axisymmetric_rigid_slab_temporal(U_0, Soln, omega,t))
        plt.legend(['Simulations', 'Theory'])
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png") 
        plt.clf()
        plt.close("all")

        
    # get dt
    dt = min(
        0.9 * dx ** 2 / 4 / nu,
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )
    if fotoTimer + dt > fotoTimer_limit:
        dt = fotoTimer_limit - fotoTimer

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, char_func, U_0 * np.cos(omega* t), 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_periodic(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx, per_communicator
    )
    vorticity[...] += penal_vorticity

    advect_vorticity_via_particles_periodic(
        z_particles, r_particles, vort_particles, vorticity, Z_double, R_double, grid_size_r, u_z, u_r, dx, dt
        )

    # diffuse vorticity
    diffusion_RK2_periodic(vorticity, temp_vorticity, R, nu, dt, dx, per_communicator)

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    freqTimer = freqTimer + dt
    if it % 50 == 0:
        print(f"time: {t}, max vort: {np.amax(vorticity)}")


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
)