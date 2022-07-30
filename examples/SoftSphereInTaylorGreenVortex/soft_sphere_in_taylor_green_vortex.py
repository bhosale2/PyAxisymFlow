import numpy as np
import matplotlib.pyplot as plt
import sys
import skfmm
import os

sys.path.append("../../")
from set_sim_params import grid_size_z, grid_size_r, dx, eps, LCFL, domain_AR
from utils.plotset import plotset
from utils.custom_cmap import lab_cmp
from utils.dump_vtk import vtk_init, vtk_write
from kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from kernels.compute_vorticity_from_velocity import compute_vorticity_from_velocity_unb
from kernels.smooth_Heaviside import smooth_Heaviside
from kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from kernels.FDM_stokes_psi_solve import (
    stokes_psi_init,
    stokes_psi_solve_LU,
)
from kernels.diffusion_RK2 import diffusion_RK2_unb
import core.particles_to_mesh as p2m
from elasto_kernels.div_tau import update_vorticity_from_solid_stress
from elasto_kernels.solid_sigma import solid_sigma
from elasto_kernels.extrapolate_eta_using_least_squares_unb import (
    extrapolate_eta_with_least_squares,
)
from elasto_kernels.advect_refmap_via_eno3 import gen_advect_refmap_via_eno3

plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
moll_zone = dx * 4
extrap_zone = moll_zone + 4 * dx
reinit_band = extrap_zone
r_ball = 0.15
rho_f = 1.0
tEnd = 2.0
freqTimer_limit = tEnd / 100
nu = 1e-3
G = 1

# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

# load initial conditions
Z_cm = 0.5
R_cm = 0.0
ball_phi = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_ball
ball_char_func = 0 * Z
smooth_Heaviside(ball_char_func, ball_phi, moll_zone)
inside_solid = ball_char_func > 0.5

vorticity = 0 * Z
penal_vorticity = 0 * Z
temp_vorticity = 0 * Z
psi = 0 * Z
u_z = 0 * Z
u_r = 0 * Z
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

eta1 = Z.copy()
eta2 = R.copy()
eta1_double = Z_double.copy()
eta2_double = R_double.copy()
u_z_double = R_double.copy()
u_r_double = R_double.copy()
ball_phi_double = 0 * Z_double

kz = 2 * np.pi
kr = kz
C = 5e-2
vorticity[...] = C * (kz**2 + kr**2) * np.sin(kz * Z) * np.sin(kr * R)

t = 0
it = 0
freqTimer = 0.0

_, _, LU_decomp_psi = stokes_psi_init(R)
vtk_image_data, temp_vtk_array, writer = vtk_init()


bad_phi = 0 * Z
phi_orig = 0 * Z
total_flux_double = 0 * R_double
sigma_s_11 = 0 * Z
sigma_s_12 = 0 * Z
sigma_s_22 = 0 * Z
eta1z = 0 * Z
eta2z = 0 * Z
eta1r = 0 * Z
eta2r = 0 * Z
tau_z = 0 * Z
tau_r = 0 * Z
advect_refmap_via_eno3 = gen_advect_refmap_via_eno3(dx, grid_size_r, grid_size_z)

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    stokes_psi_solve_LU(psi, LU_decomp_psi, vorticity, R)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    # plotting!!
    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        plt.contourf(
            Z,
            R,
            vorticity,
            levels=np.linspace(-25, 25, 25),
            extend="both",
            cmap=lab_cmp,
        )
        plt.colorbar()
        plt.contour(Z, R, inside_solid * eta1, levels=20, cmap="Greens", linewidths=2)
        plt.contour(Z, R, inside_solid * eta2, levels=20, cmap="Purples", linewidths=2)
        plt.contour(
            Z,
            R,
            ball_char_func,
            levels=[
                0.5,
            ],
            colors="k",
        )
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()

    # get dt
    dt = min(
        LCFL * dx / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
        LCFL * dx / np.sqrt(G / rho_f),
        0.9 * dx**2 / 4 / nu,
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if t + dt > tEnd:
        dt = tEnd - t

    advect_refmap_via_eno3(
        eta1,
        eta2,
        u_z,
        u_r,
        dt,
    )

    # pin eta and phi boundary
    phi_orig[...] = -np.sqrt((eta1 - Z_cm) ** 2 + (eta2 - R_cm) ** 2) + r_ball
    band = np.where(ball_phi > -3 * dx)  # assuming solid_CFL << 1
    ball_phi[band] = phi_orig[band]

    # reinit level set
    bad_phi[...] = ball_phi
    ball_phi = skfmm.distance(ball_phi, dx=dx, narrow=reinit_band)
    idx = ball_phi.mask
    ball_phi[idx] = bad_phi[idx]

    # get char function
    ball_char_func *= 0
    smooth_Heaviside(ball_char_func, ball_phi, moll_zone)
    inside_solid[...] = ball_char_func > 0.5

    # extrapolate eta for stresses
    extrapolate_eta_with_least_squares(
        inside_solid,
        ball_phi,
        eta1,
        eta2,
        ball_phi_double,
        eta1_double,
        eta2_double,
        extrap_zone,
        grid_size_r,
        z,
    )

    # compute solid stresses and blend
    solid_sigma(
        sigma_s_11,
        sigma_s_12,
        sigma_s_22,
        G,
        dx,
        eta1,
        eta2,
        eta1z,
        eta1r,
        eta2z,
        eta2r,
    )
    sigma_s_11[...] = ball_char_func * sigma_s_11
    sigma_s_12[...] = ball_char_func * sigma_s_12
    sigma_s_22[...] = ball_char_func * sigma_s_22
    update_vorticity_from_solid_stress(
        vorticity, tau_z, tau_r, sigma_s_11, sigma_s_12, sigma_s_22, R, dt, dx
    )

    # advect particles
    advect_vorticity_via_particles(
        z_particles,
        r_particles,
        vort_particles,
        vorticity,
        Z_double,
        R_double,
        grid_size_r,
        u_z,
        u_r,
        dx,
        dt,
    )

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    #  update time
    t += dt
    freqTimer += dt
    it += 1
    if it % 100 == 0:
        print(t, np.amax(vorticity))
os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
