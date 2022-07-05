from numba import njit
from set_sim_params import fastmath_flag, parallel_flag


@njit(cache=True, fastmath=fastmath_flag, parallel=parallel_flag)
def compute_velocity_from_phi_unb(u_z, u_r, phi, dx):
    """
    computes velocity from the Stokes stream function
    """
    # u_z = d_phi / dz
    u_z[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    u_z[:, 0] = (-phi[:, 2] + 4 * phi[:, 1] - 3 * phi[:, 0]) / (2 * dx)
    u_z[:, -1] = (phi[:, -3] - 4 * phi[:, -2] + 3 * phi[:, -1]) / (2 * dx)

    # u_r = d_phi / dr
    u_r[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
    u_r[0, :] = (-phi[2, :] + 4 * phi[1, :] - 3 * phi[0, :]) / (2 * dx)
    u_r[-1, :] = (phi[-3, :] - 4 * phi[-2, :] + 3 * phi[-1, :]) / (2 * dx)
