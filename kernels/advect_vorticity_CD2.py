from numba import njit, prange


# @njit(fastmath=True)
@njit(parallel=True, fastmath=True)
def advect_vorticity_CD2(vorticity, flux, u_z, u_r, dt, dx):
    """
    central difference 2nd order advection, usually unstable but works with viscosity
    """
    flux[...] = flux * 0
    flux[1:-1, 1:-1] = dt * (
        (vorticity[1:-1, 2:] - vorticity[1:-1, :-2]) / (2 * dx) * u_z[1:-1, 1:-1]
        + (vorticity[2:, 1:-1] - vorticity[:-2, 1:-1]) / (2 * dx) * u_r[1:-1, 1:-1]
    )
    vorticity[...] = vorticity - flux


@njit(parallel=True, fastmath=True)
def advect_vorticity_CD2_numba(vorticity, u_z, u_r, dt, dx):
    r_grid_size = vorticity.shape[0]
    z_grid_size = vorticity.shape[1]
    for i in prange(1, r_grid_size - 1):
        for j in prange(1, z_grid_size - 1):
            vorticity[i, j] -= dt * (
                (vorticity[i, j + 1] - vorticity[i, j - 1]) / (2 * dx) * u_z[i, j]
                + (vorticity[i + 1, j] - vorticity[i - 1, j]) / (2 * dx) * u_r[i, j]
            )
