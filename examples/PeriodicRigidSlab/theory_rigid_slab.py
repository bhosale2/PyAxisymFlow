import numpy as np
from sympy import (
    lambdify,
    symbols,
    besselj,
)


def theory_axisymmetric_rigid_slab_spatial(omega, nu, R_tube):
    y, t = symbols("y, t", real=True)
    k1 = omega / nu
    lam1 = np.sqrt(1j * k1)
    A = 0.5 / besselj(0, lam1 * R_tube)
    func_soln = A * besselj(0, lam1 * y)
    soln_lam = lambdify(y, func_soln)
    y_range = np.linspace(0, R_tube, 100)
    spatial_soln = soln_lam(y_range)
    return spatial_soln


def theory_axisymmetric_rigid_slab_temporal(U_0, spatial_soln, omega, t):
    temp_soln = U_0 * np.real(2 * spatial_soln * np.exp(-1j * omega * (t)))
    return temp_soln
