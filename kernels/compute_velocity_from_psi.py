from numba import njit


# @njit(fastmath=True)
@njit(fastmath=True, parallel=True)
def compute_velocity_from_psi_unb(u_z, u_r, psi, R, dx):
    """
    computes velocity from the Stokes stream function
    """
    # u_z = d_psi / dr / r
    u_z[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx) / R[1:-1, :]
    u_z[0, :] = (-psi[2, :] + 4 * psi[1, :] - 3 * psi[0, :]) / (2 * dx) / R[0, :]
    u_z[-1, :] = (psi[-3, :] - 4 * psi[-2, :] + 3 * psi[-1, :]) / (2 * dx) / R[-1, :]

    # u_r = -d_psi / dz / r
    u_r[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx) / R[:, 1:-1]
    u_r[:, 0] = -(-psi[:, 2] + 4 * psi[:, 1] - 3 * psi[:, 0]) / (2 * dx) / R[:, 0]
    u_r[:, -1] = -(psi[:, -3] - 4 * psi[:, -2] + 3 * psi[:, -1]) / (2 * dx) / R[:, -1]
