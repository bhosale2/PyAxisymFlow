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

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e12
moll_zone = np.sqrt(2) * dx 
freq = 8.0
freqTimer_limit = 1 / freq
omega = 2 * np.pi * freq
lambda_part = 20.0# r_part ** 2 * omega / 3 / nu
r0_bubble = 0.25
r_part = 0.2 * r0_bubble
nu = r_part ** 2 * omega / 3.0 / lambda_part
rho_f = 1.0
e = 0.01
U_0 = e * r0_bubble * omega
no_cycles = 220
rho_s = 1*rho_f

# Build discrete domain
z = np.linspace(0 + dx / 2, 1 - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, domain_AR - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

# load initial conditions
rp = 2.0
bubble_Z_cm = 0.5 - rp * r0_bubble
bubble_R_cm = 0.0
bubble_phi = -np.sqrt((Z - bubble_Z_cm) ** 2 + (R - bubble_R_cm) ** 2) + r0_bubble
bubble_char_func = 0 * Z
smooth_Heaviside(bubble_char_func, bubble_phi, moll_zone)
inside_bubble = bubble_char_func >= 0.5

part_Z_cm = bubble_Z_cm + rp * r0_bubble
part_R_cm = 0.0
part_phi = -np.sqrt((Z - part_Z_cm) ** 2 + (R - part_R_cm) ** 2) + r_part
part_char_func = 0 * Z
smooth_Heaviside(part_char_func, part_phi, moll_zone)
part_mass = rho_s * np.sum(part_char_func * R)
part_vol = np.sum(part_char_func * R)
part_Z_cm_old = part_Z_cm

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

part_trajectory = []
avg_T = []
avg_part_trajectory = []
cycle_end_part_trajectory = []
avg_Z_cm = 0.0
avg_time = 0.0
cycle_time = 0.0

_, _, LU_decomp_psi = stokes_psi_init(R)
vtk_image_data, temp_vtk_array, writer = vtk_init()

F_proj = 0.0
M_proj = 0.0
M_proj_old = 0.0

old_dt = min(
    0.9 * dx ** 2 / 4 / nu,
    LCFL / (np.amax(np.sqrt(u_r ** 2 + u_z ** 2)) + eps),
    0.01 * freqTimer_limit,
)

temp_gradient = 0 * Z
pos_flux = 0 * Z
neg_flux = 0 * Z
mid_vorticity = 0 * Z
T_n = 0
T_o = 0
diff=0
# solver loop

# restart load
if os.path.exists("restart.npz"):
    restart = np.load("restart.npz")
    t = float(restart["t"])
    vorticity[...] = restart["vorticity"]
    part_phi[...] = restart["part_phi"]
    T = restart["T"]
    U_z_cm_part_old  = float(restart["U_z_cm_part_old"])
    U_z_cm_part = float(restart["U_z_cm_part"])
    F_pen = float(restart["F_pen"])
    F_un = float(restart["F_un"])
    part_Z_cm_new  = float(restart["part_Z_cm_new"])
    part_Z_cm = float(restart["part_Z_cm"])
    part_Z_cm_old = float(restart["part_Z_cm_old"])
    part_trajectory = restart["part_trajectory"]
    avg_T = restart["avg_T"]
    avg_part_trajectory = restart["avg_part_trajectory"]
    cycle_end_part_trajectory = restart["cycle_end_part_trajectory"]
    avg_Z_cm = float(restart["avg_Z_cm"])
    avg_time = float(restart["avg_time"])
    cycle_time = float(restart["cycle_time"])
    diff = float(restart["diff"])
    #print(t, part_Z_cm)
else:
    F_un = 0
    F_pen = 0.0
    U_z_cm_part = 0.0
    U_z_cm_part_old = 0.0
    t = 0.0
    T = []

    #print(f"Restarting at t = {t}")
tEnd =  (no_cycles / freq)
while t < tEnd:
    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
    kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

    # solve for stream function and get velocity
    stokes_psi_solve_LU(psi, LU_decomp_psi, vorticity, R)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(
            Z,
            R,
            avg_vort,
            levels=np.linspace(-0.005, 0.005, 100),
            extend="both",
            cmap=lab_cmp,
        )
        #plt.colorbar()
        plt.contour(
            Z,
            R,
            bubble_char_func,
            levels=[
                0.5,
            ],
            colors="grey",
        )
        plt.contour(
            Z,
            R,
            avg_psi,
            colors="green",
        )
        plt.contour(
            Z,
            R,
            avg_part_char_func,
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
        plt.clf()
        plt.close("all")
        vtk_write(
            "bubble_avg_" + str("%0.4d" % (t * 100)) + ".vti",
            vtk_image_data,
            temp_vtk_array,
            writer,
            ["avg_part_char_func", "avg_psi", "avg_vort","u_z","u_r","avg_psi"],
            [avg_part_char_func, avg_psi, avg_vort, u_z, u_r, avg_psi],
        )
        # avg_T.append(t - 0.5 * freqTimer_limit)
        # avg_part_trajectory.append((avg_Z_cm - bubble_Z_cm) / r0_bubble)
        avg_T = np.append(avg_T, (avg_time / cycle_time))
        avg_part_trajectory = np.append(avg_part_trajectory, (avg_Z_cm / cycle_time - bubble_Z_cm) / r0_bubble)
        cycle_end_part_trajectory = np.append(cycle_end_part_trajectory, (part_Z_cm - bubble_Z_cm) / r0_bubble)
       # avg_T.append(avg_time / cycle_time)
       # avg_part_trajectory.append((avg_Z_cm / cycle_time - bubble_Z_cm) / r0_bubble)
       # cycle_end_part_trajectory.append((part_Z_cm - bubble_Z_cm) / r0_bubble)


    # get dt
        np.savez(
                "restart.npz",
                t=t,
                vorticity=vorticity,
                part_phi=part_phi,
                T=T,
                U_z_cm_part_old = U_z_cm_part_old,
                U_z_cm_part = U_z_cm_part,
                F_pen = F_pen,
                F_un = F_un,
                part_Z_cm_new = part_Z_cm_new,
                part_Z_cm = part_Z_cm,
                part_Z_cm_old = part_Z_cm_old,
                part_trajectory = part_trajectory,
                avg_T = avg_T,
                avg_part_trajectory = avg_part_trajectory,
                cycle_end_part_trajectory = cycle_end_part_trajectory,
                avg_Z_cm = avg_Z_cm,
                avg_time = avg_time,
                cycle_time = cycle_time,
                diff = diff,
            )

        avg_part_char_func[...] *= 0.0
        avg_psi[...] *= 0.0
        avg_vort[...] *= 0.0
        avg_Z_cm = 0.0
        avg_time = 0.0
        cycle_time = 0.0

    dt = min(
        0.9 * dx ** 2 / 4 / nu,
        # LCFL / (np.amax(np.sqrt(u_r ** 2 + u_z ** 2)) + eps),
        LCFL / (np.amax(np.fabs(vorticity)) + eps),
        0.01 * freqTimer_limit,
    )


    # get bubble breathing mode
    u_z_breath[...] = U_0 * (Z - bubble_Z_cm) * np.sin(omega * t) / r0_bubble
    u_r_breath[...] = U_0 * (R - bubble_R_cm) * np.sin(omega * t) / r0_bubble

    # add potential flow (bubble)
    u_z[...] += inside_bubble * u_z_breath
    u_r[...] += inside_bubble * u_r_breath
    u_z[...] += (
        (1.0 - inside_bubble)
        * U_0
        * (Z - bubble_Z_cm)
        * np.sin(omega * t)
        * r0_bubble ** 2
        / ((Z - bubble_Z_cm) ** 2 + (R - bubble_R_cm) ** 2) ** 1.5
    )
    u_r[...] += (
        (1.0 - inside_bubble)
        * U_0
        * (R - bubble_R_cm)
        * np.sin(omega * t)
        * r0_bubble ** 2
        / ((Z - bubble_Z_cm) ** 2 + (R - bubble_R_cm) ** 2) ** 1.5
    )

    # integrate averaged fields
    avg_part_char_func[...] += part_char_func * dt / freqTimer_limit
    avg_psi[...] += psi * dt / freqTimer_limit
    avg_vort[...] += vorticity * dt / freqTimer_limit
    # avg_Z_cm += part_Z_cm * dt / freqTimer_limit
    avg_Z_cm += part_Z_cm * dt
    avg_time += t * dt
    cycle_time += dt

    # record particle location
    T = np.append(T, t)
    part_trajectory = np.append(part_trajectory,(part_Z_cm - bubble_Z_cm) / r0_bubble)
    # set body velocity fields
    part_phi[...] = -np.sqrt((Z - part_Z_cm) ** 2 + (R - part_R_cm) ** 2) + r_part
    part_char_func *= 0
    smooth_Heaviside(part_char_func, part_phi, moll_zone)

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam, dt, part_char_func, U_z_cm_part, 0.0, u_z_upen, u_r_upen, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
    )
    vorticity[...] += penal_vorticity

    

    # compute penalisation force and unsteady force

    F_pen = rho_f * brink_lam * np.sum(R * part_char_func * (u_z - U_z_cm_part))
    F_un =  (diff*part_vol) /dt
    u_z_old[...]  = u_z
    
    old_dt = dt

    # FDM CD advection, usually unstable but works for low Re
    # flux = temp_vorticity
    # advect_vorticity_CD2(vorticity, flux, u_z, u_r, dt, dx)

    # particle advection
    z_particles[grid_size_r:, :] += u_z * dt
    z_particles[:grid_size_r, :] += np.flip(u_z, axis=0) * dt
    r_particles[grid_size_r:, :] += u_r * dt
    r_particles[:grid_size_r, :] += -np.flip(u_r, axis=0) * dt

    # # remesh
    vort_particles[grid_size_r:, :] = vorticity
    vort_particles[:grid_size_r, :] = -np.flip(vorticity, axis=0)
    vort_double[...] *= 0
    p2m.particles_to_mesh_2D_unbounded_mp4(
        z_particles, r_particles, vort_particles, vort_double, dx, dx
    )
    z_particles[...] = Z_double
    r_particles[...] = R_double
    vorticity[...] = vort_double[grid_size_r:, :]

    # correct conservative advection, cancels out with vortex stretching term
    # vorticity[...] -= vorticity * dt * u_r / R

    # stretching term
    # vortex_stretching(vorticity, u_r, R, dt)

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

    # update particle location and velocity
    U_z_cm_part_old = U_z_cm_part
    U_z_cm_part += 0.5*dt*(diff/dt+ ((F_pen + F_un) / part_mass))
    diff= dt * (F_pen + F_un) / part_mass
    part_Z_cm_new = part_Z_cm
    part_Z_cm +=  U_z_cm_part_old*dt + (0.5*dt*dt*(F_pen + F_un)/ part_mass)
    part_Z_cm_old = part_Z_cm_new


    t += dt
    fotoTimer += dt
    freqTimer += dt
    it += 1
    #if it % 100 == 0:
       # print(t, np.amax(vorticity), part_Z_cm, F_pen, F_un)
        
    

np.savetxt(
    "trajectory.csv",
    np.c_[np.array(T) * omega, np.array(part_trajectory)],
    delimiter=",",
)
np.savetxt(
    "avg_trajectory.csv",
    np.c_[
        np.array(avg_T) * omega,
        np.array(avg_part_trajectory),
        np.array(cycle_end_part_trajectory),
    ],
    delimiter=",",
)
