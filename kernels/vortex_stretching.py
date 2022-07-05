from numba import njit
from set_sim_params import fastmath_flag, parallel_flag


@njit(cache=True, fastmath=fastmath_flag, parallel=parallel_flag)
def vortex_stretching(vorticity, u_r, R, dt):
    """
    update field due to vortex stretching term
    """
    vorticity[1:-1, 1:-1] += (
        dt * u_r[1:-1, 1:-1] * vorticity[1:-1, 1:-1] / R[1:-1, 1:-1]
    )
