from numba import njit


# @njit(fastmath=True)
@njit(fastmath=True, parallel=True)
def vortex_stretching(vorticity, u_r, R, dt):
    """
    update field due to vortex stretching term
    """
    vorticity[1:-1, 1:-1] += (
        dt * u_r[1:-1, 1:-1] * vorticity[1:-1, 1:-1] / R[1:-1, 1:-1]
    )
