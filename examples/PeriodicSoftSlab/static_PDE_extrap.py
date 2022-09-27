import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

from pyaxisymflow.kernels.periodic_boundary_ghost_comm import (
    gen_periodic_boundary_ghost_comm,
    gen_periodic_boundary_ghost_comm_eta,
)


class StaticPDEExtrapolation:
    def __init__(
        self,
        dx,
        grid_size_r,
        grid_size_z,
        extrap_tol,
        extrap_band,
        periodic=False,
        per_communicator_gen=None,
        per_communicator_eta=None,
    ):
        self.dx = dx
        self.extrap_tol = extrap_tol
        self.extrap_band = extrap_band

        if periodic and (not per_communicator_gen) and (not per_communicator_eta):
            raise ValueError(
                "Periodic commuicators cannot be NoneType for periodic BCs"
            )

        self.periodic = periodic
        self.per_communicator_gen = per_communicator_gen
        self.per_communicator_eta = per_communicator_eta
        self.eps = np.finfo(float).eps
        self.offset = np.sqrt(2) * dx

        # Interim variables
        self.phi = np.zeros((grid_size_r, grid_size_z))
        self.grad_eta_n = self.phi * 0

        self.inside_solid = self.phi[2:-2, 2:-2] * 0
        self.n_r = self.inside_solid * 0
        self.n_z = self.inside_solid * 0
        self.extrap_zone1 = self.inside_solid * 0
        self.extrap_zone2 = self.inside_solid * 0

        self.n_r_positive = self.inside_solid * 0
        self.n_z_positive = self.inside_solid * 0
        self.n_r_negative = self.inside_solid * 0
        self.n_z_negative = self.inside_solid * 0

        self.original = self.inside_solid * 0
        self.previous = self.inside_solid * 0

    """
    Extrapolate target field from inside the level set (phi > 0)
    up to extrap_band. The level set function is changed to negative
    inside the function

    @Params:
        eta: target field to extrapolate
        phi: level set function (distance to zero level set);
             inside solid should have phi > 0
    """

    def extrapolate(self, eta, phi):
        self.phi[...] = -phi

        if self.periodic:
            self.per_communicator_gen(self.phi)
            self.per_communicator_eta(eta)

        self._zones_setup()
        self._compute_normal_upwind()

        grad_eta_r = (eta[3:-1, 2:-2] - eta[1:-3, 2:-2]) / (2 * self.dx)
        grad_eta_z = (eta[2:-2, 3:-1] - eta[2:-2, 1:-3]) / (2 * self.dx)

        # Inside solid where eta is known: grad_eta_n = dot(n, grad_eta)
        self.grad_eta_n[2:-2, 2:-2] = self.inside_solid * (
            self.n_r * grad_eta_r + self.n_z * grad_eta_z
        )

        self.n_r_positive[...] = (self.n_r >= 0) * self.n_r
        self.n_z_positive[...] = (self.n_z >= 0) * self.n_z
        self.n_r_negative[...] = self.n_r - self.n_r_positive
        self.n_z_negative[...] = self.n_z - self.n_z_positive

        denom = 3.0 * (np.abs(self.n_r) + np.abs(self.n_z)) + self.eps

        # Jacobi iteration to solve
        # H(psi) grad_phi * grad(grad_eta_n) = 0
        self._jacobi_iterate(
            self.grad_eta_n,
            phi * 0,
            self.extrap_zone1,
            denom,
        )

        # Jacobi iteration to solve
        # H(psi) (grad_phi * grad_eta - grad_eta_n) = 0
        self._jacobi_iterate(
            eta,
            self.grad_eta_n,
            self.extrap_zone2,
            denom,
        )

    # =================================================
    # =============== Helper Functions ================
    # =================================================
    def _compute_normal_upwind(self):
        self.n_r[...] = (
            np.maximum(
                3.0 * self.phi[2:-2, 2:-2]
                - 4.0 * self.phi[1:-3, 2:-2]
                + self.phi[:-4, 2:-2],
                0,
            )
            + np.minimum(
                -3.0 * self.phi[2:-2, 2:-2]
                + 4.0 * self.phi[3:-1, 2:-2]
                - self.phi[4:, 2:-2],
                0,
            )
        ) / (2 * self.dx)

        self.n_z[...] = (
            np.maximum(
                3.0 * self.phi[2:-2, 2:-2]
                - 4.0 * self.phi[2:-2, 1:-3]
                + self.phi[2:-2, :-4],
                0,
            )
            + np.minimum(
                -3.0 * self.phi[2:-2, 2:-2]
                + 4.0 * self.phi[2:-2, 3:-1]
                - self.phi[2:-2, 4:],
                0,
            )
        ) / (2 * self.dx)

        n_mag = np.sqrt(self.n_r**2 + self.n_z**2)
        self.n_r /= n_mag + self.eps
        self.n_z /= n_mag + self.eps

    # normal * grad(soln_field) = rhs_field
    def _jacobi_iterate(self, soln_field, rhs_field, extrap_zone, denominator):
        residual = 1 + self.extrap_tol
        self.original[...] = soln_field[2:-2, 2:-2]
        self.previous[...] = soln_field[2:-2, 2:-2]
        while residual > self.extrap_tol:
            extrap_num_r = self.n_r_positive * (
                4.0 * soln_field[1:-3, 2:-2] - soln_field[:-4, 2:-2]
            ) - self.n_r_negative * (
                4.0 * soln_field[3:-1, 2:-2] - soln_field[4:, 2:-2]
            )

            extrap_num_z = self.n_z_positive * (
                4.0 * soln_field[2:-2, 1:-3] - soln_field[2:-2, :-4]
            ) - self.n_z_negative * (
                4.0 * soln_field[2:-2, 3:-1] - soln_field[2:-2, 4:]
            )
            soln_field[2:-2, 2:-2] = (
                self.original
                + extrap_zone
                * (2 * self.dx * rhs_field[2:-2, 2:-2] + extrap_num_r + extrap_num_z)
                / denominator
            )
            residual = la.norm(soln_field[2:-2, 2:-2] - self.previous)
            self.previous[...] = soln_field[2:-2, 2:-2]

    def _zones_setup(self):
        self.inside_solid[...] = self.phi[2:-2, 2:-2] <= -self.offset
        self.extrap_zone1[...] = (self.phi[2:-2, 2:-2] > -self.offset) & (
            self.phi[2:-2, 2:-2] < self.extrap_band
        )
        self.extrap_zone2[...] = (self.phi[2:-2, 2:-2] >= 0.0) & (
            self.phi[2:-2, 2:-2] < self.extrap_band
        )
