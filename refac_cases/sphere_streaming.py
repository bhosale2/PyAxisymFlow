import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
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
    stokes_psi_solve_gmres,
    stokes_psi_solve_LU,
)
# from kernels.poisson_solve_unb import pseudo_poisson_solve_unb

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
u_z_pen = 0 * Z
u_r_pen = 0 * Z
avg_psi = 0 * Z
Z_cm = 0.5
R_cm = 0.0
t = 0
fotoTimer = 0.0
it = 0
freqTimer = 0.0

#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
char_func = smooth_Heaviside(phi0, moll_zone)
d = np.ma.array(char_func, mask=char_func < 0.5)

FDM_matrix, precond_matrix, LU_decomp = stokes_psi_init(R)
vtk_image_data, temp_vtk_array, writer = vtk_init()

# solver loop
while t < tEnd:

    if freqTimer >= freqTimer_limit:
        freqTimer = 0.0
        fotoTimer = 0.0
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
        vtk_write(
            "axisym_avg_" + str("%0.4d" % (t * 100)) + ".vti",
            vtk_image_data,
            temp_vtk_array,
            writer,
            ["char_func", "avg_psi"],
            [char_func, avg_psi],
        )
        avg_psi[...] = 0 * Z

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3)
    kill_boundary_vorticity_sine_r(vorticity, R, 3)

    # solve for stream function and get velocity
    # stokes_psi_solve_gmres(psi, FDM_matrix, precond_matrix, vorticity, R)
    stokes_psi_solve_LU(psi, LU_decomp, vorticity, R)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R)

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
    char_func = smooth_Heaviside(phi, moll_zone)

    # penalise velocity
    u_z_pen[...], u_r_pen[...] = brinkmann_penalize(
        brink_lam, dt, char_func, U_0 * np.cos(omega * t), 0, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(penal_vorticity, u_z_pen - u_z, u_r_pen - u_r)
    vorticity[...] += penal_vorticity

    # stretching term
    vortex_stretching(vorticity, u_r_pen, R, dt)

    # FDM CD advection, usually unstable but works for low Re
    advect_vorticity_CD2(vorticity, u_z_pen, u_r_pen, dt, dx)

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt)

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    freqTimer = freqTimer + dt
    if it % 50 == 0:
        print(t, np.amax(vorticity))


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
