import skfmm
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyaxisymflow.utils.custom_cmap import lab_cmp

# from static_PDE_extrap import StaticPDEExtrapolation
from pyaxisymflow.kernels.bounded_static_PDE_extrapolation import StaticPDEExtrapolation
from pyaxisymflow.elasto_kernels.extrapolate_eta_using_least_squares_unb import (
    extrapolate_eta_with_least_squares,
)
from pyaxisymflow.utils.custom_cmap import lab_cmp
from extrap_LS import extrap_LS


def compute_distance(x, y, x_center=0.0, y_center=0.0):
    return np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)


def central_difference(phi, dx):
    phi_x = phi * 0
    phi_y = phi * 0
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
    phi_y[0, :] = (-3.0 * phi[0, :] + 4.0 * phi[1, :] - phi[2, :]) / (2 * dx)
    phi_y[-1, :] = (3.0 * phi[-1, :] - 4.0 * phi[-2, :] + phi[-3, :]) / (2 * dx)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    phi_x[:, 0] = (-3.0 * phi[:, 0] + 4.0 * phi[:, 1] - phi[:, 2]) / (2 * dx)
    phi_x[:, -1] = (3.0 * phi[:, -1] - 4.0 * phi[:, -2] + phi[:, -3]) / (2 * dx)
    return phi_x, phi_y


grid_size_x = 256
grid_size_y = grid_size_x
dx = 1.0 / grid_size_x
x = np.linspace(dx / 2, 1 - dx / 2, grid_size_x)
y = np.linspace(dx / 2, 1 - dx / 2, grid_size_y)
X, Y = np.meshgrid(x, y)
R = 0.2
x_cm = 0.5
y_cm = 0.5
etp = StaticPDEExtrapolation(
    dx=dx,
    grid_size_r=grid_size_y,
    grid_size_z=grid_size_x,
    extrap_tol=1e-3,
    extrap_band=0.2,
)

phi = R - compute_distance(X, Y, x_cm, y_cm)
inside = phi > 0
eta_x = X * inside
eta_y = Y * inside
# phi_x, phi_y = central_difference(-phi, dx)

phi_mask = phi * inside
etp.extrapolate(eta_x, phi)
# extrap_LS(dx, grid_size_y, grid_size_x, np.finfo(float).eps, -eta_x, phi, 1e-4, 0.3)

plt.contour(X, Y, phi, levels=[0])
plt.contourf(X, Y, eta_x, cmap=lab_cmp)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
