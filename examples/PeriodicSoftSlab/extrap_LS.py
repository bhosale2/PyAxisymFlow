import numpy as np
import numpy.linalg as la
import scipy.sparse as spp


def extrap_LS(
    dx,
    grid_size_r,
    grid_size_z,
    eps,
    eta,
    phi,
    extrap_tol,
    extrap_band,
    num_threads=False,
):

    gradient = (
        spp.diags(
            [-0.5, 0.5, -0.5, 0.5],
            [-1, 1, grid_size_r - 1, -grid_size_r + 1],
            shape=(grid_size_r, grid_size_r),
            format="csr",
        )
        / dx
    )

    gradient_T = (
        spp.diags(
            [-0.5, 0.5, -0.5, 0.5],
            [-1, 1, grid_size_z - 1, -grid_size_z + 1],
            shape=(grid_size_z, grid_size_z),
            format="csr",
        )
        / dx
    )

    shift_p = spp.diags(
        [1, 1],
        [1, -grid_size_r + 1],
        shape=(grid_size_r, grid_size_r),
        format="csr",
    )

    shift_p_T = spp.diags(
        [1, 1],
        [1, -grid_size_z + 1],
        shape=(grid_size_z, grid_size_z),
        format="csr",
    )

    shift_pp = spp.diags(
        [1, 1],
        [2, -grid_size_r + 2],
        shape=(grid_size_r, grid_size_r),
        format="csr",
    )

    shift_pp_T = spp.diags(
        [1, 1],
        [2, -grid_size_z + 2],
        shape=(grid_size_z, grid_size_z),
        format="csr",
    )

    shift_n = spp.diags(
        [1, 1],
        [-1, grid_size_r - 1],
        shape=(grid_size_r, grid_size_r),
        format="csr",
    )

    shift_n_T = spp.diags(
        [1, 1],
        [-1, grid_size_z - 1],
        shape=(grid_size_z, grid_size_z),
        format="csr",
    )

    shift_nn_T = shift_nn = spp.diags(
        [1, 1],
        [-2, grid_size_z - 2],
        shape=(grid_size_z, grid_size_z),
        format="csr",
    )

    shift_nn = spp.diags(
        [1, 1],
        [-2, grid_size_r - 2],
        shape=(grid_size_r, grid_size_r),
        format="csr",
    )

    upwind_2p = (
        spp.diags(
            [3.0, -4.0, 1.0, -4.0, 1.0],
            [0, -1, -2, grid_size_r - 1, grid_size_r - 2],
            shape=(grid_size_r, grid_size_r),
            format="csr",
        )
        / 2
        / dx
    )

    upwind_2p_T = (
        spp.diags(
            [3.0, -4.0, 1.0, -4.0, 1.0],
            [0, -1, -2, grid_size_z - 1, grid_size_z - 2],
            shape=(grid_size_z, grid_size_z),
            format="csr",
        )
        / 2
        / dx
    )

    upwind_2n = (
        spp.diags(
            [-3.0, 4.0, -1.0, 4.0, -1.0],
            [0, 1, 2, -grid_size_r + 1, -grid_size_r + 2],
            shape=(grid_size_r, grid_size_r),
            format="csr",
        )
        / 2
        / dx
    )

    upwind_2n_T = (
        spp.diags(
            [-3.0, 4.0, -1.0, 4.0, -1.0],
            [0, 1, 2, -grid_size_z + 1, -grid_size_z + 2],
            shape=(grid_size_z, grid_size_z),
            format="csr",
        )
        / 2
        / dx
    )

    """
    extrapolates a field to positive LS values
    using static PDE based extrapolation
    """
    # phi_x = phi * gradient_T
    # phi_y = gradient * phi
    # phi_x = np.maximum(phi * upwind_1p_T, 0) + np.minimum(phi * upwind_1n_T, 0)
    # phi_y = np.maximum(upwind_1p * phi, 0) + np.minimum(upwind_1n * phi, 0)
    phi_x = np.maximum(phi * upwind_2p_T, 0) + np.minimum(phi * upwind_2n_T, 0)
    phi_y = np.maximum(upwind_2p * phi, 0) + np.minimum(upwind_2n * phi, 0)
    s = np.sqrt(phi_x**2 + phi_y**2)
    phi_x /= s + eps
    phi_y /= s + eps
    eta_x = eta * gradient_T
    eta_y = gradient * eta
    eta_n = (phi + np.sqrt(2) * dx <= 0) * (phi_x * eta_x + phi_y * eta_y)
    # eta_n = (phi + 2 * np.sqrt(2) * dx <= 0) * (phi_x * eta_x + phi_y * eta_y)
    dir_xp = (phi_x >= 0) * phi_x
    dir_yp = (phi_y >= 0) * phi_y
    dir_xn = (phi_x < 0) * phi_x
    dir_yn = (phi_y < 0) * phi_y
    # denom = np.absolute(phi_x) + np.absolute(phi_y)
    denom = 1.5 * (np.absolute(phi_x) + np.absolute(phi_y))

    """
    extrapolate slope: H(psi + h) phi_n * eta_n = 0
    using second order upwind marching
    """
    H1 = (phi + np.sqrt(2) * dx) > 0
    # H1 = (phi + 2 * np.sqrt(2) * dx) > 0
    H2 = phi < extrap_band
    eta_n0 = eta_n.copy()
    eta_p = eta_n * 0.0
    res = 1
    while res > extrap_tol:
        # extrap_num_y = dir_yp * (shift_n * eta_n) - dir_yn * (shift_p * eta_n)
        # extrap_num_x = dir_xp * (eta_n * shift_n_T) - dir_xn * (eta_n * shift_p_T)
        extrap_num_y = dir_yp * (
            2 * shift_n * eta_n - 0.5 * shift_nn * eta_n
        ) - dir_yn * (2 * shift_p * eta_n - 0.5 * shift_pp * eta_n)
        extrap_num_x = dir_xp * (
            2 * eta_n * shift_n_T - 0.5 * eta_n * shift_nn_T
        ) - dir_xn * (2 * eta_n * shift_p_T - 0.5 * eta_n * shift_pp_T)
        eta_n[...] = eta_n0 + H1 * H2 * ((extrap_num_x + extrap_num_y) / (denom + eps))
        res = la.norm(H2 * (eta_n - eta_p))
        eta_p[...] = eta_n.copy()

    """
    extrapolate eta: H(psi) (phi_n * eta - eta_n) = 0
    using second order upwind marching
    """
    H1 = phi >= 0
    H2 = phi < extrap_band
    eta_0 = eta.copy()
    eta_p = eta * 0.0
    res = 1
    while res > extrap_tol:
        # extrap_num_y = dir_yp * (shift_n * eta) - dir_yn * (shift_p * eta)
        # extrap_num_x = dir_xp * (eta * shift_n_T) - dir_xn * (eta * shift_p_T)
        extrap_num_y = dir_yp * (2 * shift_n * eta - 0.5 * shift_nn * eta) - dir_yn * (
            2 * shift_p * eta - 0.5 * shift_pp * eta
        )
        extrap_num_x = dir_xp * (
            2 * eta * shift_n_T - 0.5 * eta * shift_nn_T
        ) - dir_xn * (2 * eta * shift_p_T - 0.5 * eta * shift_pp_T)
        eta[...] = eta_0 + H1 * H2 * (
            (eta_n * dx + extrap_num_x + extrap_num_y) / (denom + eps)
        )
        res = la.norm(H2 * (eta - eta_p))
        eta_p[...] = eta.copy()

    return extrap_LS
