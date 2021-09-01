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
    # stokes_psi_solve_gmres,
    stokes_psi_solve_LU,
)
# from kernels.poisson_solve_unb import pseudo_poisson_solve_unb

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
tEnd = 2.0  # End time
fotoTimer_limit = 0.2
brink_lam = 1e4
moll_zone = dx * 2 ** 0.5
U = 1.0
Re_D = 40
cyl_R = 0.075
nu = U * 2 * cyl_R / Re_D

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
Z_cm = 0.35
R_cm = 0.0
t = 0
fotoTimer = 0.0
it = 0

#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + cyl_R
# phi0 = 0.1 - np.fabs(R - 0.75)
char_func = smooth_Heaviside(phi0, moll_zone)
d = np.ma.array(char_func, mask=char_func < 0.5)

FDM_matrix, precond_matrix, LU_decomp = stokes_psi_init(R)
vtk_image_data, temp_vtk_array, writer = vtk_init()

# solver loop
while t < tEnd:

    # kill vorticity at boundaries
    kill_boundary_vorticity_sine_z(vorticity, Z, 3)
    kill_boundary_vorticity_sine_r(vorticity, R, 3)

    # solve for stream function and get velocity
    # stokes_psi_solve_gmres(psi, FDM_matrix, precond_matrix, vorticity, R)
    stokes_psi_solve_LU(psi, LU_decomp, vorticity, R)
    compute_velocity_from_psi_unb(u_z, u_r, psi, R)
    # pseudo_poisson_solve_unb(psi, vorticity, R)
    # u_z[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
    # u_z[0, :] = (-psi[2, :] + 4 * psi[1, :] - 3 * psi[0, :]) / (2 * dx)
    # u_z[-1, :] = (psi[-3, :] - 4 * psi[-2, :] + 3 * psi[-1, :]) / (2 * dx)

    # # u_r = -d_psi / dz / r
    # u_r[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)
    # u_r[:, 0] = -(-psi[:, 2] + 4 * psi[:, 1] - 3 * psi[:, 0]) / (2 * dx)
    # u_r[:, -1] = -(psi[:, -3] - 4 * psi[:, -2] + 3 * psi[:, -1]) / (2 * dx)

    # add free stream
    u_z[...] += U

    # get dt
    dt = min(0.9 * dx ** 2 / 4 / nu, LCFL / (np.amax(np.fabs(vorticity)) + eps))

    # penalise velocity
    u_z_pen[...], u_r_pen[...] = brinkmann_penalize(
        brink_lam, dt, char_func, 0, 0, u_z, u_r
    )
    compute_vorticity_from_velocity_unb(penal_vorticity, u_z_pen - u_z, u_r_pen - u_r)
    vorticity[...] += penal_vorticity

    # stretching term
    vortex_stretching(vorticity, u_r_pen, R, dt)

    # FDM CD advection, usually unstable but works for low Re
    advect_vorticity_CD2(vorticity, u_z_pen, u_r_pen, dt)

    # diffuse vorticity
    diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt)

    #  update time
    t += dt
    fotoTimer += dt
    it += 1
    if it % 50 == 0:
        print(t, np.amax(vorticity))

    # Plot solution
    if fotoTimer > fotoTimer_limit or t >= tEnd:
        fotoTimer = 0.0
        levels = np.linspace(-0.1, 0.1, 25)
        plt.contourf(Z, R, u_r_pen, levels=100, extend="both", cmap=lab_cmp)
        plt.colorbar()
        plt.contour(
            Z,
            R,
            u_z_pen,
            levels=[
                0.0,
            ],
            colors="red",
        )
        plt.contourf(Z, R, d, cmap="Greys", zorder=2)
        plt.gca().set_aspect("equal")
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()
        vtk_write(
            "axisym_" + str("%0.4d" % (t * 100)) + ".vti",
            vtk_image_data,
            temp_vtk_array,
            writer,
            ["char_func", "vort", "u_z", "u_r"],
            [char_func, vorticity, u_z, u_r],
        )

os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
)
