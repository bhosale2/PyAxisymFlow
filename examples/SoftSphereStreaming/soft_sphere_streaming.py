import numpy as np
import matplotlib.pyplot as plt
import skfmm
import os
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_unb
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3
from pyaxisymflow.elasto_kernels.div_tau import update_vorticity_from_solid_stress
from pyaxisymflow.elasto_kernels.solid_sigma import solid_sigma
from pyaxisymflow.elasto_kernels.extrapolate_eta_using_least_squares_unb import (
    extrapolate_eta_with_least_squares,
)
from pyaxisymflow.elasto_kernels.advect_refmap_via_eno3 import (
    gen_advect_refmap_via_eno3,
)


# global settings
grid_size_z = 256
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
LCFL = 0.1
eps = np.finfo(float).eps
num_threads = 4

plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_moll_zone = dx * 2**0.5
brink_lam = 1e8
moll_zone = dx * 2
extrap_zone = moll_zone + 4 * dx
r_ball = 0.15
reinit_tol = 1e-5
reinit_band = extrap_zone
rho_f = 1
rho_s = rho_f
freq = 16
freqTimer_limit = 1 / freq
# freqTimer_limit = 0.01
omega = 2 * np.pi * freq
nond_AC = 0.125
e = 0.1
U_0 = e * r_ball * omega
Rs = (e / nond_AC) ** 2
nu = e * U_0 * r_ball / Rs
no_cycles = 30
tEnd = no_cycles / freq
Cauchy = 0.05 * 2
G = e * rho_f * (r_ball * omega) ** 2 / Cauchy
zeta = 0.25

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
fixed_rad = zeta * r_ball
tether_phi = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + fixed_rad
tether_char_func = 0 * Z
smooth_Heaviside(tether_char_func, tether_phi, moll_zone)

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
u_z_double = R_double.copy()
u_r_double = R_double.copy()
ball_phi_double = 0 * Z_double

avg_psi = 0 * Z
avg_phi = 0 * Z

t = 0
it = 0
freqTimer = 0.0

advect_refmap_via_eno3 = gen_advect_refmap_via_eno3(dx, grid_size_r, grid_size_z)
FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)


temp_gradient = 0 * Z
pos_flux = 0 * Z
neg_flux = 0 * Z
mid_vorticity = 0 * Z
bad_phi = 0 * Z
phi_orig = 0 * Z
temp_gradient = 0 * Z
total_flux = 0 * Z
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
eta1_0 = eta1.copy()
eta2_0 = eta2.copy()
inside_solid0 = inside_solid.copy()


# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    # plotting!!
    if freqTimer >= 1 * freqTimer_limit:
        freqTimer = 0.0
        plt.contour(Z, R, -avg_psi, levels=10, extend="both", cmap="Greys")
        plt.contourf(Z, R, -avg_psi, levels=50, extend="both", cmap=lab_cmp)
        plt.colorbar()
        plt.contour(
            Z, R, inside_solid0 * eta1_0, levels=20, cmap="Greens", linewidths=2
        )
        plt.contour(
            Z, R, inside_solid0 * eta2_0, levels=20, cmap="Purples", linewidths=2
        )
        plt.contour(
            Z,
            R,
            avg_phi,
            levels=[
                0.0,
            ],
            colors="k",
        )
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()

        avg_psi[...] *= 0
        avg_phi[...] *= 0

    # get dt
    dt = min(
        LCFL * dx / np.sqrt(G / rho_f),
        0.9 * dx**2 / 4 / nu,
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
    )
    if freqTimer + dt > freqTimer_limit:
        dt = freqTimer_limit - freqTimer
    if t + dt > tEnd:
        dt = tEnd - t

    # integrate averaged fields
    avg_psi[...] += psi * dt
    avg_phi[...] += ball_phi * dt

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

    # advect vorticity
    advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

    # get char function
    ball_char_func = 0 * Z
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

    Z_cm_t = Z_cm + e * r_ball * np.sin(omega * t)
    tether_phi[...] = -np.sqrt((Z - Z_cm_t) ** 2 + (R - R_cm) ** 2) + fixed_rad
    tether_char_func = 0 * Z
    smooth_Heaviside(tether_char_func, tether_phi, moll_zone)

    # Brinkman_Penalization
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam,
        dt,
        tether_char_func,
        U_0 * np.cos(omega * (t)),
        0.0,
        u_z_upen,
        u_r_upen,
        u_z,
        u_r,
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    )
    vorticity[...] += penal_vorticity

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
