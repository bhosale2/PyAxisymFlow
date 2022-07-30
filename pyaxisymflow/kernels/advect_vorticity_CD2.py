from numba import njit


@njit(cache=True)
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
