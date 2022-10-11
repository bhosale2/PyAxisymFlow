import numpy as np
import os
from pyaxisymflow.utils.dump_vtk import vtk_init, vtk_write
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
from bounded_static_PDE_extrapolation import StaticPDEExtrapolation
from pyaxisymflow.elasto_kernels.div_tau import (
    update_vorticity_from_solid_stress_periodic,
)
from pyaxisymflow.elasto_kernels.solid_sigma import solid_sigma_periodic
from pyaxisymflow.elasto_kernels.advect_refmap_via_eno3 import (
    gen_advect_refmap_via_eno3_periodic,
)
from theory_soft_slab import (
    theory_axisymmetric_soft_slab_spatial,
)
from periodic_soft_slab_post_processing import (
    plot_velocity_profile_with_theory,
    plot_vorticity_contours,
)


def simualte_periodic_soft_slab(
    grid_size_r,
    Re,
    Er,
    domain_AR=32,
    zeta=1.0,
    match_resolution=False,
    compare_with_theory=True,
    plot_contour=True,
    save_vtk=False,
):
    # Build discrete domain
    max_r = 0.5
    max_z = max_r / domain_AR
    grid_size_z = int(grid_size_r / domain_AR)
    dx = max_r / grid_size_r
    z = np.linspace(0 + dx / 2, max_z - dx / 2, grid_size_z)
    r = np.linspace(0 + dx / 2, max_r - dx / 2, grid_size_r)
    Z, R = np.meshgrid(z, r)

    # Build periodic communicators
    ghost_size = 2
    per_communicator1 = gen_periodic_boundary_ghost_comm(ghost_size)
    per_communicator2 = gen_periodic_boundary_ghost_comm_eta(ghost_size, max_z, dx)

    # Global parameters
    CFL = 0.1
    eps = np.finfo(float).eps
    brink_lam = 1e4
    moll_zone = np.sqrt(2) * dx
    extrap_zone = moll_zone + 3 * dx
    reinit_band = extrap_zone
    extrap_tol = 1e-3

    # Geometric parameters
    wall_thickness_half = 0.05
    R_wall_center = (
        0.5 * max_r
    )  # oscillating wall is located half way in the radial domain
    R_tube = R_wall_center - wall_thickness_half
    L = 2 * R_tube
    L_f = R_tube * zeta / (1 + zeta)
    L_s = R_tube / (1 + zeta)

    freq = 1
    freqTimer_limit = 0.1 / freq
    omega = 2 * np.pi * freq
    freqTimer = freqTimer_limit

    # Non-dimensional params:
    # shear_rate = 2 * V_wall / (omega * L)
    # zeta = L_f / L_s
    # Re = shear_rate * omega * L_f ** 2 / nu_f
    # Er = mu_f * shear_rate * omega / G
    #
    # We assume fluid and solid have the same density and viscosity
    V_wall = 1.0
    rho_f = 1.0
    shear_rate = 2 * V_wall / omega / L
    nu_f = shear_rate * omega * L_f**2 / Re
    G = rho_f * nu_f * shear_rate * omega / Er

    # Set simulation time
    nondim_T = 15
    tEnd = nondim_T / freq
    T_ramp = tEnd / 15.0

    # load initial conditions
    solid_phi = L_s - R
    solid_char_func = 0 * Z
    smooth_Heaviside(solid_char_func, solid_phi, moll_zone)
    inside_solid = solid_char_func > 0.5
    wall_phi = -np.sqrt((R - R_wall_center) ** 2) + wall_thickness_half
    wall_char_func = 0 * Z
    smooth_Heaviside(wall_char_func, wall_phi, moll_zone)

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

    t = 0
    it = 0

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

    extrapolate_refmap_via_static_pde = StaticPDEExtrapolation(
        dx=dx,
        grid_size_r=grid_size_r,
        grid_size_z=grid_size_z,
        extrap_tol=extrap_tol,
        extrap_band=extrap_zone,
        periodic=True,
        per_communicator_gen=per_communicator1,
        per_communicator_eta=per_communicator2,
    )

    if save_vtk:
        multiple_factor = 128
        domain_z_range = max_z - 2 * ghost_size * dx
        domain_z_grid_size = grid_size_z - 2 * ghost_size

        extended_solid_char_func = np.zeros(
            (grid_size_r, domain_z_grid_size * multiple_factor),
            dtype=np.float64,
        )
        extended_wall_char_func = extended_solid_char_func * 0.0
        extended_vorticity = extended_solid_char_func * 0.0
        extended_u_z = extended_solid_char_func * 0.0
        extended_eta1 = extended_solid_char_func * 0.0

        if not os.path.exists("vtk_data"):
            os.system("mkdir vtk_data")
        vtk_image_data, temp_vtk_array, writer = vtk_init(
            domain_z_grid_size * multiple_factor, grid_size_r
        )

    sim_pos = R[: int(grid_size_r * R_tube / max_r), int(grid_size_z / 2)]

    # Compute symbolized analytical solution
    if match_resolution:
        (
            theory_axisymmetric_soft_slab_temporal,
            Y,
        ) = theory_axisymmetric_soft_slab_spatial(
            L_f,
            L_s,
            shear_rate,
            omega,
            G,
            V_wall,
            rho_f,
            nu_f,
            resolution=sim_pos,
        )
    else:
        (
            theory_axisymmetric_soft_slab_temporal,
            Y,
        ) = theory_axisymmetric_soft_slab_spatial(
            L_f,
            L_s,
            shear_rate,
            omega,
            G,
            V_wall,
            rho_f,
            nu_f,
        )
    theory_pos = Y.copy()

    # Results to return
    time_history = []
    nondim_sim_pos = sim_pos / R_tube
    nondim_theory_pos = theory_pos / R_tube
    nondim_sim_vel = []
    nondim_theory_vel = []

    # solver loop
    while t < tEnd:

        # get dt
        dt = min(
            CFL * dx / np.sqrt(G / rho_f),
            0.9 * dx**2 / 4 / nu_f,
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

        if freqTimer >= freqTimer_limit:
            freqTimer = 0.0

            theory_v = theory_axisymmetric_soft_slab_temporal(t)
            sim_v = u_z[: int(grid_size_r * R_tube / max_r), int(grid_size_z / 2)]

            # Add to return parameters
            time_history.append(t * freq)
            nondim_theory_vel.append(theory_v / V_wall)
            nondim_sim_vel.append(sim_v / V_wall)

            # save vtk
            if save_vtk:
                extended_solid_char_func[...] = np.tile(
                    solid_char_func[:, ghost_size:-ghost_size], multiple_factor
                )
                extended_wall_char_func[...] = np.tile(
                    wall_char_func[:, ghost_size:-ghost_size], multiple_factor
                )
                extended_vorticity[...] = np.tile(
                    vorticity[:, ghost_size:-ghost_size], multiple_factor
                )
                extended_u_z[...] = np.tile(
                    u_z[:, ghost_size:-ghost_size], multiple_factor
                )
                for i in range(multiple_factor):
                    extended_eta1[
                        :,
                        i * domain_z_grid_size : (i + 1) * domain_z_grid_size,
                    ] = (
                        eta1[:, ghost_size:-ghost_size] + i * domain_z_range
                    )

                vtk_write(
                    "vtk_data/axisym_avg_" + str("%0.4d" % (t * 100)) + ".vti",
                    vtk_image_data,
                    temp_vtk_array,
                    writer,
                    [
                        "eta1",
                        "solid_char_func",
                        "wall_char_func",
                        "vorticity",
                        "u_z",
                    ],
                    [
                        extended_eta1,
                        extended_solid_char_func,
                        extended_wall_char_func,
                        extended_vorticity,
                        extended_u_z,
                    ],
                    domain_z_grid_size * multiple_factor,
                    grid_size_r,
                )

            # Plotting
            if compare_with_theory:
                plot_velocity_profile_with_theory(
                    sim_r=sim_pos,
                    sim_v=sim_v,
                    theory_r=theory_pos,
                    theory_v=theory_v,
                    time=t,
                    v_wall=V_wall,
                )

            if plot_contour:
                plot_vorticity_contours(
                    z_grid=Z,
                    r_grid=R,
                    vorticity=vorticity,
                    solid_eta=inside_solid * eta1,
                    solid_char_func=solid_char_func,
                    wall_char_func=wall_char_func,
                    time=t,
                )

        advect_refmap_via_eno3_periodic(
            eta1,
            eta2,
            u_z,
            u_r,
            dt,
        )

        # Velocity penalization to update wall velocity
        u_z_upen[...] = u_z
        u_r_upen[...] = u_r
        brinkmann_penalize(
            brink_lam,
            dt,
            wall_char_func,
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
        extrapolate_refmap_via_static_pde.extrapolate(eta1, solid_phi)
        extrapolate_refmap_via_static_pde.extrapolate(eta2, solid_phi)

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

        sigma_s_11[...] = solid_char_func * sigma_s_11
        sigma_s_12[...] = solid_char_func * sigma_s_12
        sigma_s_22[...] = solid_char_func * sigma_s_22

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
            vorticity, temp_vorticity, R, nu_f, dt, dx, per_communicator1
        )
        #  update time
        t += dt
        freqTimer += dt
        it += 1
        if it % 1000 == 0:
            print(f"time: {t:.4f}, maxvort: {np.amax(vorticity):.4f}")

    return {
        "time_history": np.array(time_history),
        "sim_positions": nondim_sim_pos,
        "sim_velocities": np.array(nondim_sim_vel),
        "theory_positions": nondim_theory_pos,
        "theory_velocities": np.array(nondim_theory_vel),
    }


if __name__ == "__main__":
    simualte_periodic_soft_slab(
        grid_size_r=256,
        Re=10,
        Er=0.25,
    )
