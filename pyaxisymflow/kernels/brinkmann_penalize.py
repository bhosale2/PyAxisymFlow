from numba import njit


@njit(cache=True)
def brinkmann_penalize(
    lam, dt, char_func, U_z, U_r, grid_u_z, grid_u_r, penalized_u_z, penalized_u_r
):
    """
    implicit Brinkmann penalization for velocity
    """
    penalized_u_z[...] = (grid_u_z + lam * dt * char_func * U_z) / (
        1 + lam * dt * char_func
    )
    penalized_u_r[...] = (grid_u_r + lam * dt * char_func * U_r) / (
        1 + lam * dt * char_func
    )
