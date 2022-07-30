import numpy as np
from pyaxisymflow.elasto_kernels.extrapolate_using_least_squares import (
    extrapolate_eta_using_least_squares,
)


def extrapolate_eta_with_least_squares(
    inside_solid,
    ball_phi,
    eta1,
    eta2,
    ball_phi_double,
    eta1_double,
    eta2_double,
    extrap_zone,
    grid_size_r,
    z,
):
    ball_phi_double[grid_size_r:, :] = -ball_phi
    eta1_double[grid_size_r:, :] = inside_solid * eta1
    eta2_double[grid_size_r:, :] = inside_solid * eta2
    ball_phi_double[:grid_size_r, :] = -np.flip(ball_phi, axis=0)
    eta1_double[:grid_size_r, :] = np.flip(eta1_double[grid_size_r:, :], axis=0)
    eta2_double[:grid_size_r, :] = -np.flip(eta2_double[grid_size_r:, :], axis=0)

    extrapolate_eta_using_least_squares(
        ball_phi_double, 0, extrap_zone, eta1_double, eta2_double, z, z
    )
    eta1[...] = eta1_double[grid_size_r:, :]
    eta2[...] = eta2_double[grid_size_r:, :]
