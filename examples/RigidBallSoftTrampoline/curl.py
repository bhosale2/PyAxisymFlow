import numpy as np
import scipy.sparse as spp


def curl(ax, ay, grid_size_z, grid_size_r, dx):
    """
    compute curl of the vector using CD
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

    vort = np.mat(ay) * gradient_T - gradient * np.mat(ax)
    return vort
