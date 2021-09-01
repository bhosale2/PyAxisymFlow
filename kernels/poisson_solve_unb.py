import numpy as np
from scipy.fftpack import fft2, ifft2
from set_sim_params import dx, grid_size_z, eps
import numpy.linalg as la


def kill_boundary_vorticity_sine_x(vorticity, X, width):
    """
    kills vorticity on the boundaries in a sine wave fashion
    in the given width in X direction
    """
    vorticity[:, :width] = np.sin(
        np.pi * (X[:, :width] - 0.5 * dx) / 2 / (width - 1) / dx
    ) * vorticity[:, (width - 1)].reshape(-1, 1)
    vorticity[:, -width:] = np.sin(
        np.pi * (1 - X[:, -width:] - 0.5 * dx) / 2 / (width - 1) / dx
    ) * vorticity[:, -width].reshape(-1, 1)


def kill_boundary_vorticity_sine_y(vorticity, Y, width):
    """
    kills vorticity on the boundaries in a sine wave fashion
    in the given width in Y direction
    """
    vorticity[:width, :] = (
        np.sin(np.pi * (Y[:width, :] - 0.5 * dx) / 2 / (width - 1) / dx)
        * vorticity[(width - 1), :]
    )
    vorticity[-width:, :] = (
        np.sin(np.pi * (1 - Y[-width:, :] - 0.5 * dx) / 2 / (width - 1) / dx)
        * vorticity[-width, :]
    )


grid_size = grid_size_z
x = np.linspace(0 + 0.5 * dx, 1 - 0.5 * dx, grid_size)
X, Y = np.meshgrid(x, x)
x_double = np.linspace(0, 2 - dx, 2 * grid_size)
X_double, Y_double = np.meshgrid(x_double, x_double)
GF = (
    np.log(
        np.sqrt(
            np.minimum(X_double, 2 - X_double) ** 2
            + np.minimum(Y_double, 2 - Y_double) ** 2
        )
        + eps
    )
    / 2
    / np.pi
)
# this term needs attention!
GF[0, 0] = (2 * np.log(dx / np.sqrt(np.pi)) - 1) / 4 / np.pi
fourier_GF = fft2(GF)
mirror_source = np.zeros((grid_size, grid_size))
mirror_old_psi = np.zeros((grid_size, grid_size))
mirror_psi = np.zeros((grid_size, grid_size))
mirror_dpsi_dr = np.zeros((grid_size, grid_size))


def poisson_solve_unb(src):
    """
    solves Poisson equation in 2D del^2(psi) = src on domain (0, 1)
    for unbounded domain using Greens function convolution and
    domain doubling trick (Hockney and Eastwood)
    """
    src_double = np.zeros((2 * grid_size, 2 * grid_size))
    src_double[:grid_size, :grid_size] = src
    psi = ifft2(fft2(src_double) * fourier_GF)
    return psi[:grid_size, :grid_size] * dx * dx


def pseudo_poisson_solve_unb(psi, vorticity, R, iter_tol=1e-7):
    """
    solves axisymmetric pseudo Poisson equation in 2D
    del^2(psi)  - dpsi_dr / R = -vorticity * R on
    domain (0, 1)
    for unbounded domain using Greens function
    convolution and domain doubling trick
    (Hockney and Eastwood)
    """
    source = -vorticity
    # source = -vorticity * R
    mirror_psi[grid_size // 2 :, :] = psi
    mirror_psi[: grid_size // 2, :] = -np.flip(psi, axis=0)
    mirror_source[grid_size // 2 :, :] = source
    mirror_source[: grid_size // 2, :] = -np.flip(source, axis=0)
    err = 1
    iter_no = 0

    while err > iter_tol:
        iter_no += 1
        mirror_old_psi[...] = mirror_psi
        mirror_dpsi_dr[1:-1, :] = (mirror_psi[2:, :] - mirror_psi[:-2, :]) / 2 / dx
        mirror_dpsi_dr[-1, :] = (
            (mirror_psi[-3, :] - 4 * mirror_psi[-2, :] + 3 * mirror_psi[-1, :]) / 2 / dx
        )
        mirror_dpsi_dr[grid_size // 2 :, :] /= R
        mirror_dpsi_dr[: grid_size // 2, :] = -np.flip(
            mirror_dpsi_dr[grid_size // 2 :, :], axis=0
        )
        # kill_boundary_vorticity_sine_x(mirror_dpsi_dr, X, width=3)
        # kill_boundary_vorticity_sine_y(mirror_dpsi_dr, Y, width=3)
        mirror_psi[...] = np.real(poisson_solve_unb(mirror_source + 0 * mirror_dpsi_dr))
        err = la.norm(mirror_psi - mirror_old_psi) * dx

    psi[...] = mirror_psi[grid_size // 2 :, :]
    # print(iter_no)
