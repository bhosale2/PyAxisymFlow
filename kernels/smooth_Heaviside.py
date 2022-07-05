import numpy as np
from numba import njit
from set_sim_params import fastmath_flag, parallel_flag


@njit(cache=True, fastmath=fastmath_flag, parallel=parallel_flag)
def smooth_Heaviside(H, phi, blend_w):
    """
    computes a smooth Heaviside function needed for smoothing
    """
    H[...] = H + (phi >= blend_w)
    H[...] = H + (np.fabs(phi) < blend_w) * 0.5 * (
        1 + phi / blend_w + np.sin(np.pi * phi / blend_w) / np.pi
    )
