from numba import njit


@njit(cache=True)
def solid_sigma(
    sigma_s_11,
    sigma_s_12,
    sigma_s_22,
    G,
    dx,
    eta_1,
    eta_2,
    eta_1z,
    eta_1r,
    eta_2z,
    eta_2r,
):
    eta_1z[1:-1, 1:-1] = (eta_1[1:-1, 2:] - eta_1[1:-1, :-2]) / (2 * dx)
    eta_2z[1:-1, 1:-1] = (eta_2[1:-1, 2:] - eta_2[1:-1, :-2]) / (2 * dx)
    eta_1r[1:-1, 1:-1] = (eta_1[2:, 1:-1] - eta_1[:-2, 1:-1]) / (2 * dx)
    # eta_1r[0, :] = (-eta_1[2, :] + 4 * eta_1[1, :] - 3 * eta_1[0, :]) / (2 * dx)
    eta_2r[1:-1, 1:-1] = (eta_2[2:, 1:-1] - eta_2[:-2, 1:-1]) / (2 * dx)
    # eta_2r[0, :] = (-eta_2[2, :] + 4 * eta_2[1, :] - 3 * eta_2[0, :]) / (2 * dx)

    sigma_s_12[...] = -2 * G * (eta_1z * eta_1r + eta_2z * eta_2r)
    sigma_s_11[...] = G * (eta_1r**2 + eta_2r**2 - eta_1z**2 - eta_2z**2)
    sigma_s_22[...] = -sigma_s_11
