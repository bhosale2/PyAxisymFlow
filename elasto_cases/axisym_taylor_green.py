import numpy as np
import matplotlib.pyplot as plt
import sys
import skfmm
import os

# sys.path.append("../")
sys.path.append("/Users/mattiagazzola/Desktop/comp_try/py_axisymflow")
from set_sim_params import grid_size_z, grid_size_r, dx, eps, LCFL, domain_AR
from utils.plotset import plotset
from utils.custom_cmap import lab_cmp
from utils.dump_vtk import vtk_init, vtk_write
from kernels.brinkmann_penalize import brinkmann_penalize
from kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from kernels.compute_vorticity_from_velocity import compute_vorticity_from_velocity_unb
from kernels.diffusion_RK2_unb import diffusion_RK2_unb
from kernels.smooth_Heaviside import smooth_Heaviside
from kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from kernels.vortex_stretching import vortex_stretching
from kernels.advect_vorticity_CD2 import advect_vorticity_CD2
from kernels.FDM_stokes_psi_solve import (
    stokes_psi_init,
    stokes_psi_solve_LU,
)
from kernels.diffusion_RK2_unb import diffusion_RK2_unb
import core.particles_to_mesh as p2m

from kernels.advect_CD2_ENO import advect_CD2_ENO
from elasto_kernels.div_tau import update_vorticity_from_solid_stress
from elasto_kernels.extrapolate_using_least_squares import (
    extrapolate_eta_using_least_squares,
)
from elasto_kernels.solid_sigma import solid_sigma

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e4
moll_zone = dx * 2
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
u_z_upen = 0 * Z
u_r_upen = 0 * Z
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
ball_phi_double = 0 * Z_double
kz = 2 * np.pi
kr = kz
C = 5e-2
vorticity[...] = C * (kz ** 2 + kr ** 2) * np.sin(kz * Z) * np.sin(kr * R)

t = 0
it = 0
freqTimer = 0.0

_, _, LU_decomp_psi = stokes_psi_init(R)
vtk_image_data, temp_vtk_array, writer = vtk_init()

temp_gradient = 0 * Z
pos_flux = 0 * Z
neg_flux = 0 * Z
mid_vorticity = 0 * Z
bad_phi = 0 * Z
phi_orig = 0 * Z
temp_gradient = 0 * Z
total_flux = 0 * Z
sigma_s_11 = 0 * Z
sigma_s_12 = 0 * Z
sigma_s_22 = 0 * Z
eta1z = 0 * Z
eta2z = 0 * Z
eta1r = 0 * Z
eta2r = 0 * Z
tau_z = 0 * Z
tau_r = 0 * Z

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    stokes_psi_solve_LU(psi, LU_decomp_psi, vorticity, R)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)
    # u_z[...] -= 0.1

    # plotting!!
    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        plt.contour(Z, R, -psi, levels=10, extend="both", cmap="Greys")
        plt.contourf(Z, R, -psi, levels=50, extend="both", cmap=lab_cmp)
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
        # vtk_write(
        #     "bubble_avg_" + str("%0.4d" % (t * 100)) + ".vti",
        #     vtk_image_data,
        #     temp_vtk_array,
        #     writer,
        #     ["avg_part_char_func", "avg_psi", "avg_vort"],
        #     [avg_part_char_func, avg_psi, avg_vort],
        # )

    # get dt
    dt = min(
        LCFL
        * dx
        / (np.amax(np.fabs(inside_solid * u_z) + np.fabs(inside_solid * u_r)) + eps),
        LCFL * dx / np.sqrt(G / rho_f),
        0.9 * dx ** 2 / 4 / nu,
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if t + dt > tEnd:
        dt = tEnd - t

    # extrapolate eta for advection
    # ball_phi_double[grid_size_r:, :] = -ball_phi
    # eta1_double[grid_size_r:, :] = inside_solid * eta1
    # eta2_double[grid_size_r:, :] = inside_solid * eta2
    # ball_phi_double[:grid_size_r, :] = -np.flip(ball_phi, axis=0)
    # eta1_double[:grid_size_r, :] = np.flip(inside_solid * eta1, axis=0)
    # eta2_double[:grid_size_r, :] = -np.flip(inside_solid * eta2, axis=0)
    # extrapolate_eta_using_least_squares(
    #     ball_phi_double, 0, extrap_zone, eta1_double, eta2_double, z, z
    # )
    # eta1[...] = eta1_double[grid_size_r:, :]
    # eta2[...] = eta2_double[grid_size_r:, :]

    # advect refmap
    advect_CD2_ENO(
        eta1, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx
    )
    advect_CD2_ENO(
        eta2, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx
    )
    # advect_CD2_ENO(
    #     ball_phi, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx
    # )
    # update_CD2_ENO_vec(eta1, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx)
    # update_CD2_ENO_vec(eta2, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx)
    # update_CD2_ENO_vec(ball_phi, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx)

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
    ball_phi_double[grid_size_r:, :] = -ball_phi
    eta1_double[grid_size_r:, :] = inside_solid * eta1
    eta2_double[grid_size_r:, :] = inside_solid * eta2
    ball_phi_double[:grid_size_r, :] = -np.flip(ball_phi, axis=0)
    eta1_double[:grid_size_r, :] = np.flip(inside_solid * eta1, axis=0)
    eta2_double[:grid_size_r, :] = -np.flip(inside_solid * eta2, axis=0)
    extrapolate_eta_using_least_squares(
        ball_phi_double, 0, extrap_zone, eta1_double, eta2_double, z, z
    )
    eta1[...] = eta1_double[grid_size_r:, :]
    eta2[...] = eta2_double[grid_size_r:, :]

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

    # penalise velocity (ball)
    # u_z_upen[...] = u_z.copy()
    # u_r_upen[...] = u_r.copy()
    # brinkmann_penalize(
    #     brink_lam, dt, ball_char_func, 0.0, 0.0, u_z_upen, u_r_upen, u_z, u_r
    # )
    # compute_vorticity_from_velocity_unb(
    #     penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    # )
    # vorticity[...] += penal_vorticity

    # stretching term
    # vortex_stretching(vorticity, u_r, R, dt)

    # FDM CD advection, usually unstable but works for low Re
    # flux = temp_vorticity
    # advect_vorticity_CD2(vorticity, flux, u_z, u_r, dt, dx)

    # advect particles
    z_particles[grid_size_r:, :] += u_z * dt
    z_particles[:grid_size_r, :] += np.flip(u_z, axis=0) * dt
    r_particles[grid_size_r:, :] += u_r * dt
    r_particles[:grid_size_r, :] += -np.flip(u_r, axis=0) * dt

    # remesh
    vort_particles[grid_size_r:, :] = vorticity
    vort_particles[:grid_size_r, :] = -np.flip(vorticity, axis=0)
    vort_double[...] *= 0
    p2m.particles_to_mesh_2D_unbounded_mp4(
        z_particles, r_particles, vort_particles, vort_double, dx, dx
    )
    z_particles[...] = Z_double
    r_particles[...] = R_double
    vorticity[...] = vort_double[grid_size_r:, :]

    # correct conservative advection, cancels out with vortec stretching term
    # vorticity[...] -= vorticity * dt * u_r / R

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
