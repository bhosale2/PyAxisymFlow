import numpy as np
import matplotlib.pyplot as plt
import os
from pyaxisymflow.utils.plotset import plotset
from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.utils.dump_vtk import vtk_init, vtk_write
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize
from pyaxisymflow.kernels.compute_velocity_from_psi import (
    compute_velocity_from_psi_unb,
)
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
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import (
    gen_advect_vorticity_via_eno3,
)
from pyaxisymflow.kernels.implicit_diffusion_solver import ImplicitEulerDiffusionStepper


def simulate_oscillating_spheroid(
    M_sq,
    radius,
    spheroid_AR,
    freq=16,
    grid_size_z=512,
    domain_AR=0.5,
    implicit_diffusion=False,
    plot_figure=False,
    save_vtk=False,
):
    # global settings
    dx = 1.0 / grid_size_z
    grid_size_r = int(domain_AR * grid_size_z)
    CFL = 0.1
    eps = np.finfo(float).eps
    num_threads = 4

    plotset()
    plt.figure(figsize=(5 / domain_AR, 5))
    # Parameters
    brink_lam = 1e4
    moll_zone = dx * 2**0.5

    # Spheroid geometry
    rad_a = radius  # radius along axis of symmetry and oscillation
    rad_b = rad_a / spheroid_AR
    rad_eq = (rad_b**2 * rad_a) ** (1 / 3)
    length_scale = spheroid_AR * rad_eq
    print(f"Problem length scale: {length_scale}")

    # streaming parameters
    omega = 2 * np.pi * freq
    nond_AC = 1.0 / np.sqrt(M_sq)
    e = 0.1
    U_0 = e * length_scale * omega
    Rs = (e / nond_AC) ** 2
    nu = e * U_0 * length_scale / Rs
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
    avg_psi = 0 * Z
    Z_cm = 0.5
    R_cm = 0.0
    t = 0
    fotoTimer = 0.0
    it = 0
    freqTimer = 0.0
    u_z_upen = 0 * Z
    u_r_upen = 0 * Z

    #  create char function
    phi = -np.sqrt((Z - Z_cm) ** 2 / spheroid_AR**2 + (R - R_cm) ** 2) + rad_b
    char_func = 0 * Z
    smooth_Heaviside(char_func, phi, moll_zone)

    FD_stokes_solver = FastDiagonalisationStokesSolver(grid_size_r, grid_size_z, dx)
    advect_vorticity_via_eno3 = gen_advect_vorticity_via_eno3(
        dx, grid_size_r, grid_size_z, num_threads=num_threads
    )

    # timestep related parameters
    snaps_per_cycle = 40
    fotoTimer_limit = 1 / freq / snaps_per_cycle
    freqTimer_limit = 1 / freq
    diffusion_dt_limit = dx**2 / 4 / nu
    dt_limit = min(diffusion_dt_limit, freqTimer_limit, fotoTimer_limit)

    if implicit_diffusion:
        implicit_diffusion_stepper = ImplicitEulerDiffusionStepper(
            time_step=diffusion_dt_limit,
            kinematic_viscosity=nu,
            grid_size_r=grid_size_r,
            grid_size_z=grid_size_z,
            dx=dx,
        )

    # Initialize vtk
    if save_vtk:
        os.makedirs("vtk_data", exist_ok=True)
        vtk_image_data, temp_vtk_array, writer = vtk_init(grid_size_z, grid_size_r)

    # solver loop
    while t < tEnd:

        # kill vorticity at boundaries
        kill_boundary_vorticity_sine_z(vorticity, Z, 3, dx)
        kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

        # solve for stream function and get velocity
        FD_stokes_solver.solve(solution_field=psi, rhs_field=vorticity)
        compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx)

        if freqTimer >= freqTimer_limit:
            freqTimer = 0.0

            if plot_figure:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.contourf(Z, R, avg_psi, levels=25, extend="both", cmap=lab_cmp)
                plt.contour(
                    Z,
                    R,
                    avg_psi,
                    levels=[
                        0.0,
                    ],
                    colors="grey",
                )
                plt.contour(
                    Z,
                    R,
                    char_func,
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

            if save_vtk:
                vtk_write(
                    "vtk_data/axisym_avg_" + str("%0.4d" % (t * 100)) + ".vti",
                    vtk_image_data,
                    temp_vtk_array,
                    writer,
                    ["char_func", "avg_psi"],
                    [char_func, avg_psi],
                    grid_size_z,
                    grid_size_r,
                )
            avg_psi[...] = 0 * Z

        if fotoTimer >= fotoTimer_limit:
            fotoTimer = 0.0
            # save whatever fields you want a snapshot of
            # np.savez("u_z" + str("%0.4d" % (t * 1e4)) + ".npz", t=t, uz=u_z_upen)
            # np.savez("u_r" + str("%0.4d" % (t * 1e4)) + ".npz", t=t, ur=u_r_upen)
            # np.savez("charf" + str("%0.4d" % (t * 1e4)) + ".npz", t=t, charf=char_func)

        # get dt
        if implicit_diffusion:
            dt = dt_limit
        else:
            dt = min(
                0.9 * dx**2 / 4 / nu,
                CFL / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
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
        Z_cm_t = Z_cm + e * length_scale * np.sin(omega * t)
        phi = rad_b - np.sqrt((Z - Z_cm_t) ** 2 / spheroid_AR**2 + (R - R_cm) ** 2)
        char_func = 0 * Z
        smooth_Heaviside(char_func, phi, moll_zone)
        # penalise velocity
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
        compute_vorticity_from_velocity_unb(
            penal_vorticity, -u_z_upen + u_z, -u_r_upen + u_r, dx
        )
        vorticity[...] += penal_vorticity

        # advect vorticity
        advect_vorticity_via_eno3(vorticity, u_z, u_r, dt)

        # diffuse vorticity
        # diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)
        if implicit_diffusion:
            implicit_diffusion_stepper.step(vorticity_field=vorticity, dt=dt)
        else:
            diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx)

        #  update time
        t += dt
        fotoTimer += dt
        it += 1
        freqTimer = freqTimer + dt
        if it % 50 == 0:
            print(f"time: {t}, dt: {dt}, max vort: {np.amax(vorticity)}")

    if plot_figure:
        os.system("rm -f 2D_advect.mp4")
        os.system(
            "ffmpeg -r 8 -s 3840x2160 -f image2 -pattern_type glob -i 'snap_*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' 2D_advect.mp4"
        )


if __name__ == "__main__":
    simulate_oscillating_spheroid(
        M_sq=100.2,
        radius=0.075,
        spheroid_AR=1,
        grid_size_z=256,
        plot_figure=True,
    )
