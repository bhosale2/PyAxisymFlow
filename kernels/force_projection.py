import numpy as np


def force_projection(rho_s, char_func, u_z, u_r, R):
    """
    computes projected momentum on floating bodies from the fluid
    """
    M = np.sum(rho_s * char_func * R)

    U_z = np.sum(rho_s * char_func * u_z * R) / M
    U_r = np.sum(rho_s * char_func * u_r * R) / M

    return U_z, U_r
