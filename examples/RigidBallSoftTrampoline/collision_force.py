import numpy as np
import scipy.sparse as spp


def collision_force(k, phi_1, phi_2, blend_w, grid_size_z, grid_size_r, eps, dx):
    """
    computes the collision force between 2 bodies
    using their level sets
    """

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

    phi_diff = 0.5 * (phi_2 - phi_1)
    delta_phi = phi_diff * 0
    delta_phi += (
        (np.fabs(phi_diff) < blend_w)
        * 0.5
        * (1 + np.cos(np.pi * phi_diff / blend_w))
        / blend_w
    )
    phi_diff_x = np.mat(phi_diff) * gradient_T
    phi_diff_y = gradient * np.mat(phi_diff)
    s = np.sqrt(np.square(phi_diff_x) + np.square(phi_diff_y))
    phi_diff_x /= s + eps
    phi_diff_y /= s + eps
    S = phi_diff / np.sqrt(phi_diff**2 + eps**2)
    body_mask = np.logical_or((phi_1 > 0), (phi_2 > 0))
    force_x = body_mask * S * phi_diff_x * k * delta_phi
    force_y = body_mask * S * phi_diff_y * k * delta_phi
    return force_x, force_y
