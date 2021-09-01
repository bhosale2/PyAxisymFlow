from numba import njit


# @njit(fastmath=True)
@njit(fastmath=True, parallel=True)
def compute_vorticity_from_velocity_unb(vort, u_z, u_r, dx):
    """
    compute vorticity from velocity field
    """
    # vorticity is killed at domain ends anyways so leave boundaries untouched
    # using 2nd order CDM for interior
    vort[1:-1, 1:-1] = (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx) - (
        u_z[2:, 1:-1] - u_z[:-2, 1:-1]
    ) / (2 * dx)
