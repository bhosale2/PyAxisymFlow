import numpy as np
import matplotlib.pyplot as plt
import skfmm
import os
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_forces import compute_force_on_body
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
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
from curl import curl
from collision_force import collision_force
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.utils.dump_vtk import vtk_init, vtk_write
from trampoline_level_set import trampoline_level_set

# global settings
grid_size_z = 400
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
eps = np.finfo(float).eps
num_threads = 4
implicit_diffusion = False


plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
fotoTimer_limit = 0.01
moll_zone = dx * np.sqrt(16)
brink_lam = 1e4
extrap_zone = moll_zone + 4 * dx
reinit_band = extrap_zone
scale = 1
r_ball = scale * 0.075
U_0 = 1.0
Re = 100.0
nu = 0.001
nondim_T = 3000
tEnd = 10
T_ramp = 20 * r_ball / U_0
rho_f = 1
G = 0.1
# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)
K_tether = G / (dx**2)
# load initial conditions
vorticity = 0 * Z
penal_vorticity = 0 * Z
penal_vorticity_t = 0 * Z
temp_vorticity = 0 * Z
psi = 0 * Z
u_z = 0 * Z
u_r = 0 * Z
u_z_upen = 0 * Z
u_r_upen = 0 * Z
u_z_upen_t = 0 * Z
u_r_upen_t = 0 * Z


fotoTimer = 0.0
it = 0
Z_cm1 = 0.60
Z_cm2 = 0.30
R_cm = 0.0
t = 0.0
T = []
rho_s = 1.05
g = 5 * (rho_s - rho_f) * 9.81
#  create char function
ball_phi2 = -np.sqrt((Z - Z_cm2) ** 2 + (R - R_cm) ** 2) + r_ball
ball_char_func2 = 0 * Z
smooth_Heaviside(ball_char_func2, ball_phi2, moll_zone)
inside_solid2 = ball_char_func2 > 0

# trampoline phi
trampoline_CM_Z = 0.6
trampoline_diameter = scale * 0.4
trampoline_height = scale * 0.05
left_center_R = -trampoline_diameter / 2
left_center_Z = trampoline_CM_Z
corner_radius = 0.5 * trampoline_height
ball_phi1 = -trampoline_level_set(
    R,
    Z,
    left_center_R,
    trampoline_diameter,
    trampoline_CM_Z,
    corner_radius,
    corner_radius,
)
ball_char_func1 = 0 * Z
smooth_Heaviside(ball_char_func1, ball_phi1, moll_zone)
inside_solid1 = ball_char_func1 > 0
Z_cm = trampoline_CM_Z
R_cm_t = trampoline_diameter / 2
fixed_rad = 0.01
tether_phi = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm_t) ** 2) + fixed_rad
tether_char_func = 0 * Z
smooth_Heaviside(tether_char_func, tether_phi, moll_zone)

part_mass = 2 * np.pi * dx * dx * rho_s * np.sum(ball_char_func2 * R)
part_vol = 2 * np.pi * dx * dx * np.sum(ball_char_func2 * R)
part_Z_cm_old = Z_cm2
part_Z_cm_new = Z_cm2
part_Z_cm = Z_cm2

advect_refmap_via_eno3 = gen_advect_refmap_via_eno3(dx, grid_size_r, grid_size_z)
FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)

diffusion_dt_limit = dx / np.sqrt(G)
if implicit_diffusion:
    implicit_diffusion_stepper = ImplicitEulerDiffusionStepper(
        time_step=diffusion_dt_limit,
        kinematic_viscosity=nu,
        grid_size_r=grid_size_r,
        grid_size_z=grid_size_z,
        dx=dx,
    )
r1 = np.linspace(0 + dx / 2, domain_AR - dx / 2, 2 * grid_size_r)
Z_double, R_double = np.meshgrid(z, r1)
F_un = 0
F_pen = 0.0
U_z_cm_part = 0
U_z_cm_part_old = 0
diff = 0
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
eta1 = Z.copy()
eta2 = R.copy()
eta1_double = Z_double.copy()
eta2_double = R_double.copy()
ball_phi_double = 0 * Z_double
part_Z_cm_old = Z_cm2
part_Z_cm_new = Z_cm2
part_Z_cm = Z_cm2
F_un = 0
F_pen = 0.0
U_z_cm_part = 0
U_z_cm_part_old = 0
diff = 0
T = []
U_z_cm = []
vtk_image_data, temp_vtk_array, writer = vtk_init(grid_size_z, grid_size_r)

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
            ball_char_func1,
            levels=[
                0.5,
            ],
            colors="grey",
        )
        plt.contour(
            Z,
            R,
            ball_char_func2,
            levels=[
                0.5,
            ],
            colors="grey",
        )
        plt.contour(
            Z,
            R,
            tether_char_func,
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
        vtk_write(
            "axisym_avg_" + str("%0.4d" % (t * 100)) + ".vti",
            vtk_image_data,
            temp_vtk_array,
            writer,
            ["ball_char_func1", "ball_char_func2", "vorticity", "psi", "u_z", "u_r"],
            [ball_char_func1, ball_char_func2, vorticity, psi, u_z, u_r],
            grid_size_z,
            grid_size_r,
        )

        # get dt
        if implicit_diffusion:
            # technically we can set any higher dt here lower than the CFL limit
            dt = diffusion_dt_limit
        else:
            dt = min(
                CFL * dx / np.sqrt(G / rho_f),
                0.9 * dx**2 / 4 / nu,
                CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
            )

    # advect refmap
    advect_refmap_via_eno3(
        eta1,
        eta2,
        u_z,
        u_r,
        dt,
    )

    compute_vorticity_from_velocity_unb(
        penal_vorticity_t,
        tether_char_func * (eta1 - Z),
        tether_char_func * (eta2 - R),
        dx,
    )
    vorticity[...] += K_tether * dt * penal_vorticity_t

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, ball_char_func2, U_z_cm_part, 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    )
    vorticity[...] += penal_vorticity

    F_pen, F_un = compute_force_on_body(
        R, ball_char_func2, rho_f, brink_lam, u_z, U_z_cm_part, part_vol, dt, diff
    )

    ball_phi2 = -np.sqrt((Z - part_Z_cm) ** 2 + (R - R_cm) ** 2) + r_ball
    ball_char_func2 = 0 * Z
    smooth_Heaviside(ball_char_func2, ball_phi2, moll_zone)

    # pin eta and phi boundary
    phi_orig[...] = -trampoline_level_set(
        eta2,
        eta1,
        left_center_R,
        trampoline_diameter,
        trampoline_CM_Z,
        corner_radius,
        corner_radius,
    )
    band = np.where(ball_phi1 > -3 * dx)  # assuming solid_CFL << 1
    ball_phi1[band] = phi_orig[band]

    # reinit level set
    bad_phi[...] = ball_phi1
    ball_phi1 = skfmm.distance(ball_phi1, dx=dx, narrow=reinit_band)
    idx = ball_phi1.mask
    idx_one = np.logical_and(idx, bad_phi < 0.0)
    idx_two = np.logical_and(idx, bad_phi > 0.0)
    ball_phi1[idx_one] = -99.0
    ball_phi1[idx_two] = 99.0

    # get char function
    ball_char_func1 = 0 * Z
    smooth_Heaviside(ball_char_func1, ball_phi1, moll_zone)
    inside_solid1[...] = ball_phi1 > 0.0

    # extrapolate eta for stresses
    extrapolate_eta_with_least_squares(
        inside_solid1,
        ball_phi1,
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

    sigma_s_11[...] = ball_char_func1 * sigma_s_11
    sigma_s_12[...] = ball_char_func1 * sigma_s_12
    sigma_s_22[...] = ball_char_func1 * sigma_s_22

    update_vorticity_from_solid_stress(
        vorticity, tau_z, tau_r, sigma_s_11, sigma_s_12, sigma_s_22, R, dt, dx
    )

    # Calculate collision forces
    fx, fy = collision_force(
        1 * G, ball_phi2, ball_phi1, 2 * moll_zone, grid_size_z, grid_size_r, eps, dx
    )
    vorticity[...] += dt * curl(fx, 0 * fy, grid_size_z, grid_size_r, dx)

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    # advect vorticity
    advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

    # update particle location and velocity
    F_total = 2 * np.pi * dx * dx * F_pen + F_un - np.sum(R * ball_char_func2 * fx)
    U_z_cm_part_old = U_z_cm_part
    U_z_cm_part += dt * ((g / rho_s + F_total / part_mass))
    diff = dt * (g / rho_s + F_total / part_mass)
    part_Z_cm_new = part_Z_cm
    Z_cm = part_Z_cm
    part_Z_cm += U_z_cm_part_old * dt + (
        0.5 * dt * dt * (g / rho_s + F_total / part_mass)
    )
    part_Z_cm_old = part_Z_cm_new

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    if it % 50 == 0:
        print(f"time: {t}, max vort: {np.amax(vorticity)},simulations:{U_z_cm_part}")


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
    "-vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
