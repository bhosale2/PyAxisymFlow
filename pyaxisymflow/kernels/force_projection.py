import numpy as np
from numba import njit


@njit(cache=True)
def force_projection(rho_s, char_func, u_z, u_r, R):
    """
    computes projected momentum on floating bodies from the fluid
    """
    # M = np.sum(rho_s * char_func * R)

    U_z = np.sum(rho_s * char_func * u_z * R) / np.sum(rho_s * char_func * R)
    U_r = np.sum(rho_s * char_func * u_r * R) / np.sum(rho_s * char_func * R)

    return U_z, U_r
