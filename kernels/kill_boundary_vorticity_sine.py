import numpy as np


def kill_boundary_vorticity_sine_z(vorticity, Z, width, dx):
    """
    kills vorticity on the boundaries in a sine wave fashion
    in the given width in Z direction
    """
    vorticity[:, :width] = np.sin(
        np.pi * (Z[:, :width] - 0.5 * dx) / 2 / (width - 1) / dx
    ) * vorticity[:, (width - 1)].reshape(-1, 1)
    vorticity[:, -width:] = np.sin(
        np.pi * (1 - Z[:, -width:] - 0.5 * dx) / 2 / (width - 1) / dx
    ) * vorticity[:, -width].reshape(-1, 1)


def kill_boundary_vorticity_sine_r(vorticity, R, width, dx):
    """
    kills vorticity on the boundaries in a sine wave fashion
    in the given width in R direction (only on maxR)
    on axis, set vorticity = 0
    """
    vorticity[-width:, :] = (
        np.sin(np.pi * (1 - R[-width:, :] - 0.5 * dx) / 2 / (width - 1) / dx)
        * vorticity[-width, :]
    )
    vorticity[0, :] = 0.0
