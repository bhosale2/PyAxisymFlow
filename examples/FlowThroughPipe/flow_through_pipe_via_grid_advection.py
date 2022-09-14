import numpy as np
import matplotlib.pyplot as plt
import os

from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_velocity_from_psi import (
    compute_velocity_from_psi_periodic,
)
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_periodic,
)
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
)
from pyaxisymflow.kernels.periodic_boundary_ghost_comm import (
    gen_periodic_boundary_ghost_comm,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_periodic
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import (
    gen_advect_vorticity_via_eno3_periodic,
)

# global settings
num_threads = 4
CFL = 0.1
eps = np.finfo(float).eps

# Build discrete domain
max_z = 0.1
max_r = 0.5
grid_size_z = 20
domain_AR = max_r / max_z
grid_size_r = int(domain_AR * grid_size_z)
dx = max_r / grid_size_r
z = np.linspace(0 + dx / 2, max_z - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, max_r - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

# Parameters
fotoTimer_limit = 0.1
brink_lam = 1e12
moll_zone = dx * 2**0.5
wall_thickness = 0.05
R_wall_center = 0.5
R_tube = R_wall_center - wall_thickness
Re = 100.0
U_0 = 1.0
R_extent = 0.5
U_act = U_0 * (R_extent / R_tube) ** 2

nu = U_act * 2 * R_tube / Re
nondim_T = 300
tEnd = nondim_T * R_tube / U_act
T_ramp = 20 * R_tube / U_act
ghost_size = 2
per_communicator = gen_periodic_boundary_ghost_comm(ghost_size)

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

t = 0.0
T = []
diff = 0
F_total = 0
#  create char function
phi0 = -np.sqrt((R - R_wall_center) ** 2) + wall_thickness
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)

psi_inner = psi[..., ghost_size:-ghost_size].copy()
FD_stokes_solver = FastDiagonalisationStokesSolver(
    grid_size_r,
    grid_size_z - 2 * ghost_size,
    dx,
    bc_type="homogenous_dirichlet_along_r_and_periodic_along_z",
)
advect_vorticity_via_eno3_periodic = gen_advect_vorticity_via_eno3_periodic(
    dx, grid_size_r, grid_size_z, per_communicator, num_threads=num_threads
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
    u_z[...] += U_0

    if fotoTimer >= fotoTimer_limit or t == 0:
        fotoTimer = 0.0
        levels = np.linspace(-0.1, 0.1, 25)
        plt.plot(R[:, int(grid_size_z / 2)], u_z[:, int(grid_size_z / 2)])
        plt.plot(
            R[:, int(grid_size_z / 2)],
            2 * U_act * (1 - (R[:, int(grid_size_z / 2)] / R_tube) ** 2),
        )
        plt.ylim([0, 2.5 * U_act])
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()
        plt.close("all")

    # get dt
    dt = min(
        0.9 * dx**2 / 4 / nu,
        CFL * dx / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
        0.01 * freqTimer_limit,
    )

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(brink_lam, dt, char_func, 0.0, 0.0, u_z_upen, u_r_upen, u_z, u_r)
    compute_vorticity_from_velocity_periodic(
        penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx, per_communicator
    )
    vorticity[...] += penal_vorticity

    advect_vorticity_via_eno3_periodic(vorticity, u_z, u_r, dt)

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
    "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' -vcodec "
    "libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
)
