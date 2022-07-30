import numpy as np
import matplotlib.pyplot as plt
import os
from pyaxisymflow.utils.plotset import plotset
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
)
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.periodic_boundary_ghost_comm import (
    gen_periodic_boundary_ghost_comm,
)
from pyaxisymflow.kernels.compute_velocity_from_psi import (
    compute_velocity_from_psi_periodic,
)
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_periodic,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_periodic
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3


# global settings
grid_size_z = 256
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
eps = np.finfo(float).eps
num_threads = 4

plotset()
plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
fotoTimer_limit = 0.1
brink_lam = 1e12
moll_zone = dx * 2**0.5
r_cyl = 0.075
U_0 = 1.0
Re = 100.0
nu = U_0 * 2 * r_cyl / Re
nondim_T = 300
tEnd = nondim_T * r_cyl / U_0
T_ramp = 20 * r_cyl / U_0
ghost_size = 2
per_communicator = gen_periodic_boundary_ghost_comm(ghost_size)

# Build discrete domain
# Note that only interior of this domain (removing ghost size zone) will be used for the problem
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
u_z_upen = 0 * Z
u_r_upen = 0 * Z
fotoTimer = 0.0
it = 0
freqTimer = 0.0
freq = 8.0
freqTimer_limit = 1 / freq
Z_cm = 0.85
R_cm = 0.0
t = 0.0
T = []
#  create char function
phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_cyl
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)

# solving poisson only on the inner grid
psi_inner = psi[..., ghost_size:-ghost_size].copy()
FD_stokes_solver = FastDiagonalisationStokesSolver(
    grid_size_r,
    grid_size_z - 2 * ghost_size,
    dx,
    bc_type="homogenous_neumann_along_r_and_periodic_along_z",
)
advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, num_threads=num_threads
)

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
        levels = np.linspace(-0.1, 0.1, 25)
        plt.contourf(
            Z[..., ghost_size:-ghost_size],
            R[..., ghost_size:-ghost_size],
            vorticity[..., ghost_size:-ghost_size],
            levels=100,
            extend="both",
            cmap=lab_cmp,
        )

        plt.contour(
            Z[..., ghost_size:-ghost_size],
            R[..., ghost_size:-ghost_size],
            u_z[..., ghost_size:-ghost_size],
            levels=[
                0.0,
            ],
            colors="red",
        )

        plt.contour(
            Z[..., ghost_size:-ghost_size],
            R[..., ghost_size:-ghost_size],
            char_func[..., ghost_size:-ghost_size],
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
        0.9 * dx**2 / 4 / nu,
        CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
    )

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(brink_lam, dt, char_func, 0.0, 0.0, u_z_upen, u_r_upen, u_z, u_r)
    compute_vorticity_from_velocity_periodic(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx, per_communicator
    )
    vorticity[...] += penal_vorticity

    advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

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
    "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' -vcodec"
    " libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
)
