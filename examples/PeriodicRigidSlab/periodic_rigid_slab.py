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
from theory_rigid_slab import (
    theory_axisymmetric_rigid_slab_temporal,
    theory_axisymmetric_rigid_slab_spatial,
)


# global settings
num_threads = 4
CFL = 0.1
eps = np.finfo(float).eps

# Build discrete domain
max_z = 0.05
max_r = 0.5
grid_size_z = 16
domain_AR = max_r / max_z
grid_size_r = int(domain_AR * grid_size_z)
dx = max_r / grid_size_r
z = np.linspace(0 + dx / 2, max_z - dx / 2, grid_size_z)
r = np.linspace(0 + dx / 2, max_r - dx / 2, grid_size_r)
Z, R = np.meshgrid(z, r)

plt.figure(figsize=(5 / domain_AR, 5))
# Parameters
brink_lam = 1e4
moll_zone = dx * 2**0.5
wall_thickness = 0.05
R_wall_center = 0.3
R_tube = R_wall_center - wall_thickness
Re = 25
U_0 = 1.0
R_extent = 0.5
nu = U_0 * 2 * R_tube / Re
nondim_T = 300
tEnd = nondim_T * R_tube / U_0
T_ramp = 20 * R_tube / U_0
freq = 8.0
omega = 2 * np.pi * freq
ghost_size = 2
per_communicator = gen_periodic_boundary_ghost_comm(ghost_size)
fotoTimer_limit = 0.001

y_range = np.linspace(0, R_tube, 100)
spatial_soln = theory_axisymmetric_rigid_slab_spatial(omega, nu, R_tube)

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

t = 0.0
T = []
#  create char function
phi0 = -np.sqrt((R - R_wall_center) ** 2) + wall_thickness
char_func = 0 * Z
smooth_Heaviside(char_func, phi0, moll_zone)

psi_inner = psi[..., ghost_size:-ghost_size].copy()
FD_stokes_solver = FastDiagonalisationStokesSolver(
    grid_size_r,
    grid_size_z - 2 * ghost_size,
    dx,
    bc_type="homogenous_neumann_along_r_and_periodic_along_z",
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

    if fotoTimer >= fotoTimer_limit or t == 0:
        fotoTimer = 0.0
        levels = np.linspace(-0.1, 0.1, 25)
        plt.plot(R[:, int(grid_size_z / 2)], u_z[:, int(grid_size_z / 2)])
        plt.plot(
            y_range,
            theory_axisymmetric_rigid_slab_temporal(U_0, spatial_soln, omega, t),
        )
        plt.ylim([-U_0, U_0])
        plt.legend(["Simulations", "Theory"])
        plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
        plt.clf()
        plt.close("all")

    # get dt
    dt = min(
        0.9 * dx**2 / 4 / nu,
        CFL * dx / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
    )

    # penalise velocity (particle)
    u_z_upen[...] = u_z.copy()
    u_r_upen[...] = u_r.copy()
    brinkmann_penalize(
        brink_lam,
        dt,
        char_func,
        U_0 * np.cos(omega * t),
        0.0,
        u_z_upen,
        u_r_upen,
        u_z,
        u_r,
    )
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
    if it % 50 == 0:
        print(f"time: {t}, max vort: {np.amax(vorticity)}")


os.system("rm -f 2D_advect.mp4")
os.system(
    "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' "
    "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
)
