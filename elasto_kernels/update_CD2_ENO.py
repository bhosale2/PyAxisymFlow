from numba import njit


@njit(parallel=True)
def update_CD2_ENO_vec(
    eta, u_z, u_r, temp_gradient, total_flux, pos_flux, neg_flux, dt, dx
):
    """
    Combined CD and ENO with forward Euler step
    refer to Rycroft, 2012, JCP for details
    """
    total_flux[...] = 0 * total_flux
    temp_gradient[:, 1:-1] = (eta[:, 2:] + eta[:, :-2] - 2 * eta[:, 1:-1]) / (2 * dx)
    pos_flux[:, 1:-1] = (u_z[:, 1:-1] > 0) * (
        temp_gradient[:, 1:-1] > temp_gradient[:, :-2]
    )
    neg_flux[:, 1:-1] = (u_z[:, 1:-1] < 0) * (
        temp_gradient[:, 1:-1] > temp_gradient[:, 2:]
    )
    total_flux[:, 1:-1] -= (
        dt
        * u_z[:, 1:-1]
        * (
            pos_flux[:, 1:-1] * (eta[:, 1:-1] - eta[:, :-2]) / dx
            + neg_flux[:, 1:-1] * (eta[:, 2:] - eta[:, 1:-1]) / dx
            + (1 - pos_flux[:, 1:-1])
            * (1 - neg_flux[:, 1:-1])
            * (eta[:, 2:] - eta[:, :-2])
            / (2 * dx)
        )
    )

    temp_gradient[1:-1, :] = (eta[2:, :] + eta[:-2, :] - 2 * eta[1:-1, :]) / (2 * dx)
    pos_flux[1:-1, :] = (u_r[1:-1, :] > 0) * (
        temp_gradient[1:-1, :] > temp_gradient[:-2, :]
    )
    neg_flux[1:-1, :] = (u_r[1:-1, :] < 0) * (
        temp_gradient[1:-1, :] > temp_gradient[2:, :]
    )
    total_flux[1:-1, :] -= (
        dt
        * u_r[1:-1, :]
        * (
            pos_flux[1:-1, :] * (eta[1:-1, :] - eta[:-2, :]) / dx
            + neg_flux[1:-1, :] * (eta[2:, :] - eta[1:-1, :]) / dx
            + (1 - pos_flux[1:-1, :])
            * (1 - neg_flux[1:-1, :])
            * (eta[2:, :] - eta[:-2, :])
            / (2 * dx)
        )
    )
    eta[...] = eta + total_flux
