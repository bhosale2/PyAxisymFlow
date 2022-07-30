import numpy as np

from pyaxisymflow.pyst_kernels.advection_timestep import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel,
)


def gen_advect_vorticity_via_eno3(
    dx, grid_size_r, grid_size_z, real_t=np.float64, num_threads=False
):
    advection_timestep_eno3_kernel = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel(
            real_t, num_threads, fixed_grid_size=(2 * grid_size_r, grid_size_z)
        )
    )
    vorticity_double = np.zeros((2 * grid_size_r, grid_size_z))
    u_double = np.zeros((2, 2 * grid_size_r, grid_size_z))
    advection_flux = np.zeros_like(vorticity_double)

    def advect_vorticity_via_eno3(
        vorticity,
        u_z,
        u_r,
        dt,
    ):

        # print(u_double[0, grid_size_r:, :].shape)
        u_double[0, grid_size_r:, :] = u_z
        u_double[0, :grid_size_r, :] = np.flip(u_z, axis=0)
        u_double[1, grid_size_r:, :] = u_r
        u_double[1, :grid_size_r, :] = -np.flip(u_r, axis=0)
        vorticity_double[grid_size_r:, :] = vorticity
        vorticity_double[:grid_size_r, :] = -np.flip(vorticity, axis=0)

        advection_timestep_eno3_kernel(
            field=vorticity_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )

        vorticity[...] = vorticity_double[grid_size_r:, :]

    return advect_vorticity_via_eno3


def gen_advect_vorticity_via_eno3_periodic(
    dx,
    grid_size_r,
    grid_size_z,
    per_communicator,
    real_t=np.float64,
    num_threads=False,
):
    advection_timestep_eno3_kernel = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel(
            real_t, num_threads, fixed_grid_size=(2 * grid_size_r, grid_size_z)
        )
    )
    vorticity_double = np.zeros((2 * grid_size_r, grid_size_z))
    u_double = np.zeros((2, 2 * grid_size_r, grid_size_z))
    advection_flux = np.zeros_like(vorticity_double)

    def advect_vorticity_via_eno3_periodic(
        vorticity,
        u_z,
        u_r,
        dt,
    ):
        per_communicator(u_z)
        per_communicator(u_r)
        per_communicator(vorticity)

        # print(u_double[0, grid_size_r:, :].shape)
        u_double[0, grid_size_r:, :] = u_z
        u_double[0, :grid_size_r, :] = np.flip(u_z, axis=0)
        u_double[1, grid_size_r:, :] = u_r
        u_double[1, :grid_size_r, :] = -np.flip(u_r, axis=0)
        vorticity_double[grid_size_r:, :] = vorticity
        vorticity_double[:grid_size_r, :] = -np.flip(vorticity, axis=0)

        advection_timestep_eno3_kernel(
            field=vorticity_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )

        vorticity[...] = vorticity_double[grid_size_r:, :]

    return advect_vorticity_via_eno3_periodic
