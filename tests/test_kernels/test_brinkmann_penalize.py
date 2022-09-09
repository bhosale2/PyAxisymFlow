import pytest
import numpy as np
from pyaxisymflow.kernels.brinkmann_penalize import brinkmann_penalize


@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalize(n_values):
    lam = 2.0
    dt = 3.0
    char_func = 4.0 * np.ones((n_values, n_values))
    U_z = 1.0
    U_r = 2.0
    grid_u_z = np.zeros_like(char_func)
    grid_u_r = np.zeros_like(char_func)
    penalized_u_z = np.zeros_like(char_func)
    penalized_u_r = np.zeros_like(char_func)

    brinkmann_penalize(
        lam, dt, char_func, U_z, U_r, grid_u_z, grid_u_r, penalized_u_z, penalized_u_r
    )
    penalize_denom = 1.0 + 2.0 * 3.0 * 4.0  # 1 + lam * dt * char_func
    correct_penalized_u_z = lam * dt * U_z * char_func / penalize_denom
    correct_penalized_u_r = lam * dt * U_r * char_func / penalize_denom

    np.testing.assert_allclose(penalized_u_z, correct_penalized_u_z)
    np.testing.assert_allclose(penalized_u_r, correct_penalized_u_r)
