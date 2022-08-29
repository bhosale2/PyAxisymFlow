import numpy as np
import os

from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_velocity_from_psi import compute_velocity_from_psi_unb
from pyaxisymflow.kernels.compute_vorticity_from_velocity import (
    compute_vorticity_from_velocity_unb,
)
from pyaxisymflow.kernels.smooth_Heaviside import smooth_Heaviside
from pyaxisymflow.kernels.kill_boundary_vorticity_sine import (
    kill_boundary_vorticity_sine_r,
    kill_boundary_vorticity_sine_z,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_unb
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import gen_advect_vorticity_via_eno3
from pyaxisymflow.utils.dump_vtk import vtk_init, vtk_write
from post_processing import (
    plot_contours,
)
from pyaxisymflow.kernels.implicit_diffusion_solver import ImplicitEulerDiffusionStepper


def simulate_flow_past_sphere(
    Re,
    grid_size_z=512,
    domain_AR=0.5,
    sample_size=30,
    drag_diff=1e-5,
    implicit_diffusion=True,
    CONVERGE_DRAG=False,
    PLOT_FIGURE=False,
    SAVE_VTK=False,
):
    # global settings
    dx = 1.0 / grid_size_z
    grid_size_r = int(domain_AR * grid_size_z)
    CFL = 0.1
    eps = np.finfo(float).eps
    num_threads = 4

    # Parameters
    brink_lam = 1e12
    moll_zone = dx * 2**0.5
    r_sph = 0.1
    U_0 = 1.0
    nu = U_0 * 2 * r_sph / Re
    nondim_T = 300
    tEnd = nondim_T * r_sph / U_0
    T_ramp = 20 * r_sph / U_0
    freqTimer_limit = tEnd / 200
    freqTimer = 0.0

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
    u_z_upen = 0 * Z
    u_r_upen = 0 * Z

    it = 0
    Z_cm = 0.25
    R_cm = 0.0
    t = 0.0

    last_drags = []
    previous_mean_drag = 0

    #  create char function
    phi0 = -np.sqrt((Z - Z_cm) ** 2 + (R - R_cm) ** 2) + r_sph
    char_func = 0 * Z
    smooth_Heaviside(char_func, phi0, moll_zone)

    FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
    advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
        dx, grid_size_r, grid_size_z, num_threads=num_threads
    )

    diffusion_dt_limit = dx**2 / 4 / nu
    if implicit_diffusion:
        implicit_diffusion_stepper = ImplicitEulerDiffusionStepper(
            time_step=diffusion_dt_limit,
            kinematic_viscosity=nu,
            grid_size_r=grid_size_r,
            grid_size_z=grid_size_z,
            dx=dx,
        )

    # Initialize vtk
    if SAVE_VTK:
        if not os.path.exists("vtk_data"):
            os.system("mkdir vtk_data")
        vtk_image_data, temp_vtk_array, writer = vtk_init(grid_size_z, grid_size_r)

    # solver loop
    while t < tEnd:

        # kill vorticity at boundaries
        kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
        kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

        # solve for stream function and get velocity
        FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
        compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

        # add free stream
        prefac_x = 1.0
        if t < T_ramp:
            prefac_x = np.sin(0.5 * np.pi * t / T_ramp)
        u_z[...] += U_0 * prefac_x

        # plotting!!
        if freqTimer >= freqTimer_limit or t == 0:
            freqTimer = 0.0

            if PLOT_FIGURE:
                plot_contours(t, Z, R, vorticity, u_z, char_func, Re)

            if SAVE_VTK:
                vtk_write(
                    "vtk_data/axisym_avg_"
                    + str("%0.4d" % (t * 100))
                    + f"_Re{Re}"
                    + ".vti",
                    vtk_image_data,
                    temp_vtk_array,
                    writer,
                    ["char_func", "vorticity", "psi", "u_z", "u_r"],
                    [char_func, vorticity, psi, u_z, u_r],
                    grid_size_z,
                    grid_size_r,
                )

        # get dt
        if implicit_diffusion:
            dt = diffusion_dt_limit
        else:
            dt = min(
                0.9 * dx**2 / 4 / nu,
                CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
            )

        # penalise velocity (particle)
        u_z_upen[...] = u_z.copy()
        u_r_upen[...] = u_r.copy()
        brinkmann_penalize(
            brink_lam, dt, char_func, 0.0, 0.0, u_z_upen, u_r_upen, u_z, u_r
        )
        compute_vorticity_from_velocity_unb(
            penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx
        )
        vorticity[...] += penal_vorticity

        Cd = (
            2
            * 2
            * np.pi
            * dx
            * dx
            * brink_lam
            * np.sum(R * char_func * u_z)
            / (np.pi * r_sph**2)
        )  # (Cd = F/(0.5*p*U^2*A))

        advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

        # diffuse vorticity
        if implicit_diffusion:
            implicit_diffusion_stepper.step(vorticity_field=vorticity, dt=dt)
        else:
            diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

        #  update time
        freqTimer += dt
        t += dt
        it += 1

        if it % 100 == 0:
            print(f"time: {t}, max vort: {np.amax(vorticity)}, drag coeff: {Cd}")

        # Save drag
        last_drags.append(Cd)
        if CONVERGE_DRAG and (it % sample_size == 0):
            curr_mean_drag = sum(last_drags) / sample_size
            last_drags = []

            if previous_mean_drag == 0:  # Ignore first set
                previous_mean_drag = curr_mean_drag
                continue

            diff = abs(curr_mean_drag - previous_mean_drag)
            previous_mean_drag = curr_mean_drag

            if diff < drag_diff:
                break

    return previous_mean_drag


if __name__ == "__main__":
    simulate_flow_past_sphere(100, grid_size_z=256, PLOT_FIGURE=True, SAVE_VTK=True)
    os.system("rm -f 2D_advect.mp4")
    os.system(
        "ffmpeg -r 20 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p  2D_advect.mp4"
    )
