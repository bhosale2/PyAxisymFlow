import numpy as np

from pyaxisymflow.pyst_kernels.advection_timestep import (
    gen_advection_timestep_euler_forward_non_conservative_eno3_pyst_kernel,
)


def gen_advect_refmap_via_eno3(
    dx, grid_size_r, grid_size_z, real_t=np.float64, num_threads=False
):
    advection_timestep_eno3_kernel = (
        gen_advection_timestep_euler_forward_non_conservative_eno3_pyst_kernel(
            real_t, num_threads, fixed_grid_size=(2 * grid_size_r, grid_size_z)
        )
    )
    refmap_double = np.zeros((2 * grid_size_r, grid_size_z))
    u_double = np.zeros((2, 2 * grid_size_r, grid_size_z))
    advection_flux = np.zeros_like(refmap_double)

    def advect_refmap_via_eno3(
        eta1,
        eta2,
        u_z,
        u_r,
        dt,
    ):
        u_double[0, grid_size_r:, :] = u_z
        u_double[0, :grid_size_r, :] = np.flip(u_z, axis=0)
        u_double[1, grid_size_r:, :] = u_r
        u_double[1, :grid_size_r, :] = -np.flip(u_r, axis=0)

        # eta1 advection
        refmap_double[grid_size_r:, :] = eta1
        refmap_double[:grid_size_r, :] = np.flip(eta1, axis=0)
        advection_timestep_eno3_kernel(
            field=refmap_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )
        eta1[...] = refmap_double[grid_size_r:, :]

        # eta2 advection
        refmap_double[grid_size_r:, :] = eta2
        refmap_double[:grid_size_r, :] = -np.flip(eta2, axis=0)
        advection_timestep_eno3_kernel(
            field=refmap_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )
        eta2[...] = refmap_double[grid_size_r:, :]

    return advect_refmap_via_eno3


def gen_advect_refmap_via_eno3_periodic(
    dx,
    grid_size_r,
    grid_size_z,
    per_communicator1,
    per_communicator2,
    real_t=np.float64,
    num_threads=False,
):
    advection_timestep_eno3_kernel = (
        gen_advection_timestep_euler_forward_non_conservative_eno3_pyst_kernel(
            real_t, num_threads, fixed_grid_size=(2 * grid_size_r, grid_size_z)
        )
    )
    refmap_double = np.zeros((2 * grid_size_r, grid_size_z))
    u_double = np.zeros((2, 2 * grid_size_r, grid_size_z))
    advection_flux = np.zeros_like(refmap_double)

    def advect_refmap_via_eno3_periodic(
        eta1,
        eta2,
        u_z,
        u_r,
        dt,
    ):

        per_communicator1(u_z)
        per_communicator1(u_r)
        per_communicator2(eta1)
        per_communicator1(eta2)

        u_double[0, grid_size_r:, :] = u_z
        u_double[0, :grid_size_r, :] = np.flip(u_z, axis=0)
        u_double[1, grid_size_r:, :] = u_r
        u_double[1, :grid_size_r, :] = -np.flip(u_r, axis=0)

        # eta1 advection
        refmap_double[grid_size_r:, :] = eta1
        refmap_double[:grid_size_r, :] = np.flip(eta1, axis=0)
        advection_timestep_eno3_kernel(
            field=refmap_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )
        eta1[...] = refmap_double[grid_size_r:, :]

        # eta2 advection
        refmap_double[grid_size_r:, :] = eta2
        refmap_double[:grid_size_r, :] = -np.flip(eta2, axis=0)
        advection_timestep_eno3_kernel(
            field=refmap_double,
            advection_flux=advection_flux,
            velocity=u_double,
            dt_by_dx=(dt / dx),
        )
        eta2[...] = refmap_double[grid_size_r:, :]

    return advect_refmap_via_eno3_periodic
