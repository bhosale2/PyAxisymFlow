from numba import njit


@njit(cache=True)
def compute_vorticity_from_velocity_unb(vort, u_z, u_r, dx):
    """
    compute vorticity from velocity field
    """
    # vorticity is killed at domain ends anyways so leave boundaries untouched
    # using 2nd order CDM for interior
    vort[1:-1, 1:-1] = (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx) - (
        u_z[2:, 1:-1] - u_z[:-2, 1:-1]
    ) / (2 * dx)


@njit(cache=True)
def compute_vorticity_from_velocity_periodic(vort, u_z, u_r, dx, per_communicator):
    """
    compute vorticity from velocity field
    """
    # vorticity is killed at domain ends anyways so leave boundaries untouched
    # using 2nd order CDM for interior

    per_communicator(u_r)
    per_communicator(u_z)
    vort[1:-1, 1:-1] = (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx) - (
        u_z[2:, 1:-1] - u_z[:-2, 1:-1]
    ) / (2 * dx)
