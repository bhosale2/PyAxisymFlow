from numba import njit


@njit(parallel=True)
def advect_vorticity_maccormack(vorticity, mid_vorticity, flux, u_z, u_r, dt, dx):
    """
    MacCormack scheme
    """
    flux[1:-1, 1:-1] = (
        dt
        * (
            vorticity[1:-1, 2:] * u_z[1:-1, 2:]
            - vorticity[1:-1, 1:-1] * u_z[1:-1, 1:-1]
            + vorticity[2:, 1:-1] * u_r[2:, 1:-1]
            - vorticity[1:-1, 1:-1] * u_r[1:-1, 1:-1]
        )
        / dx
    )
    mid_vorticity[...] = vorticity - flux
    flux[1:-1, 1:-1] = (
        dt
        * (
            -mid_vorticity[1:-1, :-2] * u_z[1:-1, :-2]
            + mid_vorticity[1:-1, 1:-1] * u_z[1:-1, 1:-1]
            - mid_vorticity[:-2, 1:-1] * u_r[:-2, 1:-1]
            + mid_vorticity[1:-1, 1:-1] * u_r[1:-1, 1:-1]
        )
        / dx
    )
    vorticity[...] = 0.5 * (vorticity + mid_vorticity - flux)
