import numpy as np
import matplotlib.pyplot as plt
import os
from pyaxisymflow.utils.custom_cmap import lab_cmp
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
from pyaxisymflow.kernels.advect_vorticity_via_eno3 import (
    gen_advect_vorticity_via_eno3_periodic,
)
from pyaxisymflow.kernels.FastDiagonalisationStokesSolver import (
    FastDiagonalisationStokesSolver,
)
from pyaxisymflow.kernels.diffusion_RK2 import diffusion_RK2_periodic
from pyaxisymflow.kernels.periodic_boundary_ghost_comm import (
    gen_periodic_boundary_ghost_comm,
    gen_periodic_boundary_ghost_comm_eta,
)
from pyaxisymflow.kernels import bounded_static_PDE_extrapolation
from pyaxisymflow.elasto_kernels.div_tau import (
    update_vorticity_from_solid_stress_periodic,
)
from pyaxisymflow.elasto_kernels.solid_sigma import solid_sigma_periodic
from pyaxisymflow.elasto_kernels.advect_refmap_via_eno3 import (
    gen_advect_refmap_via_eno3_periodic,
)
from theory_soft_slab import (
    theory_axisymmetric_soft_slab_spatial,
    theory_axisymmetric_soft_slab_temporal,
)


def simualte_periodic_soft_slab(grid_size_r, Re, Er, domain_AR=32):
    # Build discrete domain
    max_r = 0.5
    max_z = max_r / domain_AR
    grid_size_z = int(grid_size_r / domain_AR)
    dx = max_r / grid_size_r
    z = np.linspace(0 + dx / 2, max_z - dx / 2, grid_size_z)
    r = np.linspace(0 + dx / 2, max_r - dx / 2, grid_size_r)
    Z, R = np.meshgrid(z, r)
    CFL = 0.1
    eps = np.finfo(float).eps

    # Parameters
    brink_lam = 1e4
    moll_zone = np.sqrt(2) * dx
    extrap_zone = moll_zone + 3 * dx
    reinit_band = extrap_zone
    wall_thickness = 0.05
    R_wall_center = 0.25
    R_tube = R_wall_center - wall_thickness
    extrap_tol = 1e-3
    U_0 = 1.0
    R_extent = 0.5
    nondim_T = 300
    tEnd = nondim_T * R_tube / U_0
    T_ramp = 20 * R_tube / U_0
    freqTimer = 0.0
    freq = 1
    freqTimer_limit = 0.1 / freq
    omega = 2 * np.pi * freq
    V_wall = 1
    L = 0.4
    L_f = 0.1
    L_s = 0.1
    shear_rate = 2 * V_wall / omega / L
    nu = shear_rate * omega * L_f**2 / Re
    ghost_size = 2
    per_communicator1 = gen_periodic_boundary_ghost_comm(ghost_size)
    per_communicator2 = gen_periodic_boundary_ghost_comm_eta(ghost_size, max_z, dx)
    G = nu * shear_rate * omega / Er
    rho_f = 1
    r_ball = 0.1
    # Build discrete domain
    y_range = np.linspace(0, R_tube, 100)

    # load initial conditions
    ball_phi = r_ball - R
    ball_char_func = 0 * Z
    smooth_Heaviside(ball_char_func, ball_phi, moll_zone)
    inside_solid = ball_char_func > 0.5
    phi0 = -np.sqrt((R - R_wall_center) ** 2) + wall_thickness
    char_func0 = 0 * Z
    smooth_Heaviside(char_func0, phi0, moll_zone)

    vorticity = 0 * Z
    penal_vorticity = 0 * Z
    temp_vorticity = 0 * Z
    psi = 0 * Z
    u_z = 0 * Z
    u_r = 0 * Z
    u_z_upen = 0 * Z
    u_r_upen = 0 * Z
    eta1 = Z.copy()
    eta2 = R.copy()
    t = 0
    it = 0

    Y, vel_sl, vel_fl = theory_axisymmetric_soft_slab_spatial(
        L_f, L_s, Re, shear_rate, omega, G, V_wall
    )

    bad_phi = 0 * Z
    phi_orig = 0 * Z
    sigma_s_11 = 0 * Z
    sigma_s_12 = 0 * Z
    sigma_s_22 = 0 * Z
    eta1z = 0 * Z
    eta2z = 0 * Z
    eta1r = 0 * Z
    eta2r = 0 * Z
    tau_z = 0 * Z
    tau_r = 0 * Z
    psi_inner = psi[..., ghost_size:-ghost_size].copy()
    FD_stokes_solver = FastDiagonalisationStokesSolver(
        grid_size_r,
        grid_size_z - 2 * ghost_size,
        dx,
        bc_type="homogenous_neumann_along_r_and_periodic_along_z",
    )
    advect_refmap_via_eno3_periodic = gen_advect_refmap_via_eno3_periodic(
        dx, grid_size_r, grid_size_z, per_communicator1, per_communicator2
    )

    advect_vorticity_via_eno3_periodic = gen_advect_vorticity_via_eno3_periodic(
        dx, grid_size_r, grid_size_z, per_communicator1
    )

    extrapolate_refmap_via_static_pde = bounded_static_PDE_extrapolation(
        dx=dx,
        grid_size_r=grid_size_r,
        grid_size_z=grid_size_z,
        extrap_tol=extrap_tol,
        extrap_band=extrap_zone,
        periodic=True,
        per_communicator_gen=per_communicator1,
        per_communicator_eta=per_communicator2,
    )

    # solver loop
    while t < tEnd:

        # get dt
        dt = min(
            CFL * dx / np.sqrt(G / rho_f),
            0.9 * dx**2 / 4 / nu,
            CFL * dx / (np.amax(np.fabs(u_z) + np.fabs(u_r)) + eps),
        )
        if freqTimer + dt > freqTimer_limit:
            dt = freqTimer_limit - freqTimer
        if t + dt > tEnd:
            dt = tEnd - t

        # kill vorticity at boundaries
        kill_boundary_vorticity_sine_r(vorticity, R, 3, dx)

        # solve for stream function and get velocity
        psi_inner[...] = psi[..., ghost_size:-ghost_size]
        FD_stokes_solver.solve(
            solution_field=psi_inner, rhs_field=vorticity[:, ghost_size:-ghost_size]
        )
        psi[..., ghost_size:-ghost_size] = psi_inner
        compute_velocity_from_psi_periodic(u_z, u_r, psi, R, dx, per_communicator1)

        # plotting!!
        if freqTimer >= freqTimer_limit:
            freqTimer = 0.0

            plt.plot(
                R[: int(grid_size_r / 2), int(grid_size_z / 2)],
                u_z[: int(grid_size_r / 2), int(grid_size_z / 2)],
            )
            plt.ylim([-V_wall, V_wall])
            vel_comb = theory_axisymmetric_soft_slab_temporal(Y, t, L_s, vel_sl, vel_fl)
            plt.scatter(Y, vel_comb, linewidth=3)
            plt.legend(["Simulation", "Theory"])
            plt.xlabel("R")
            plt.ylabel("U_z")
            plt.savefig("snap_" + str("%0.4d" % (t * 100)) + ".png")
            plt.clf()
            plt.contourf(
                Z,
                R,
                vorticity,
                levels=np.linspace(-25, 25, 25),
                extend="both",
                cmap=lab_cmp,
            )
            plt.colorbar()
            plt.contour(
                Z, R, inside_solid * eta1, levels=20, cmap="Greens", linewidths=2
            )
            plt.contour(
                Z,
                R,
                ball_char_func,
                levels=[
                    0.5,
                ],
                colors="k",
            )
            plt.contour(
                Z,
                R,
                char_func0,
                levels=[
                    0.5,
                ],
                colors="k",
            )
            plt.savefig("vort_" + str("%0.4d" % (t * 100)) + ".png")
            plt.clf()

        # eta1_old[...] = eta1
        advect_refmap_via_eno3_periodic(
            eta1,
            eta2,
            u_z,
            u_r,
            dt,
        )

        u_z_upen[...] = u_z
        u_r_upen[...] = u_r
        brinkmann_penalize(
            brink_lam,
            dt,
            char_func0,
            V_wall * np.cos(omega * t),
            0.0,
            u_z_upen,
            u_r_upen,
            u_z,
            u_r,
        )

        compute_vorticity_from_velocity_periodic(
            penal_vorticity, u_z - u_z_upen, u_r - u_r_upen, dx, per_communicator1
        )
        vorticity[...] += penal_vorticity

        eta1[...] = inside_solid * eta1
        eta2[...] = inside_solid * eta2
        extrapolate_refmap_via_static_pde.extrapolate(eta1, ball_phi)
        extrapolate_refmap_via_static_pde.extrapolate(eta2, ball_phi)

        # compute solid stresses and blend
        solid_sigma_periodic(
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
            per_communicator1,
            per_communicator2,
        )

        sigma_s_11[...] = ball_char_func * sigma_s_11
        sigma_s_12[...] = ball_char_func * sigma_s_12
        sigma_s_22[...] = ball_char_func * sigma_s_22

        update_vorticity_from_solid_stress_periodic(
            vorticity,
            tau_z,
            tau_r,
            sigma_s_11,
            sigma_s_12,
            sigma_s_22,
            R,
            dt,
            dx,
            per_communicator1,
        )

        # grid based vorticity advection
        advect_vorticity_via_eno3_periodic(vorticity, u_z, u_r, dt)

        # diffuse vorticity
        diffusion_RK2_periodic(
            vorticity, temp_vorticity, R, nu, dt, dx, per_communicator1
        )
        #  update time
        t += dt
        freqTimer += dt
        it += 1
        if it % 100 == 0:
            print(f"time: {t:.4f}, maxvort: {np.amax(vorticity):.4f}")


if __name__ == "__main__":
    simualte_periodic_soft_slab(grid_size_r=256, Re=10, Er=0.25)
