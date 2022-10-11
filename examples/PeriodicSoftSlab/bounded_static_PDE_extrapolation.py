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
        self.grid_size_r = grid_size_r
        self.grid_size_z = grid_size_z
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

        # Original matricies:             grid_size_r x grid_size_z
        # Bounded matricies:        (r_end - r_start) x (z_end - z_start)
        # Interim matricies:    (r_end - r_start - 4) x (z_end - z_start - 4)
        self.r_start = 0
        self.r_end = grid_size_r
        self.z_start = 0
        self.z_end = grid_size_z

        self.r_start_at_boundary = True
        self.r_end_at_boundary = True
        self.z_start_at_boundary = True
        self.z_end_at_boudary = True

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
        if self.periodic:
            self.per_communicator_gen(phi)
            self.per_communicator_eta(eta)

        # Find a bounding box that bounds the solid and extrapolation zone
        box_zone = phi + self.extrap_band >= 0
        self._find_bounding_box(box_zone)

        # Slice phi and eta within the bounding box
        # phi is inverted to ensure that inside solid is upwind
        bounded_phi = -phi[self.r_start : self.r_end, self.z_start : self.z_end].copy()
        bounded_eta = eta[self.r_start : self.r_end, self.z_start : self.z_end].copy()

        # Setup extrapolation zones
        inside_solid, extrap_zone = self._zones_setup(bounded_phi)

        # Compute normal vectors (grad phi / |grad phi|)
        n_r, n_z = self._compute_normal_upwind(bounded_phi)

        grad_eta_r = (bounded_eta[3:-1, 2:-2] - bounded_eta[1:-3, 2:-2]) / (2 * self.dx)
        grad_eta_z = (bounded_eta[2:-2, 3:-1] - bounded_eta[2:-2, 1:-3]) / (2 * self.dx)

        # Inside solid where eta is known: grad_eta_n = dot(n, grad_eta)
        grad_eta_n = bounded_eta * 0
        grad_eta_n[2:-2, 2:-2] = inside_solid * (n_r * grad_eta_r + n_z * grad_eta_z)

        n_r_positive = (n_r >= 0) * n_r
        n_z_positive = (n_z >= 0) * n_z
        n_r_negative = n_r - n_r_positive
        n_z_negative = n_z - n_z_positive

        denom = 3.0 * (np.abs(n_r) + np.abs(n_z)) + self.eps

        # Jacobi iteration to solve
        # H(psi) grad_phi * grad(grad_eta_n) = 0
        self._jacobi_iterate(
            grad_eta_n,
            bounded_phi * 0,
            extrap_zone,
            inside_solid,
            denom,
            n_r_positive,
            n_r_negative,
            n_z_positive,
            n_z_negative,
        )

        # Jacobi iteration to solve
        # H(psi) (grad_phi * grad_eta - grad_eta_n) = 0
        self._jacobi_iterate(
            bounded_eta,
            grad_eta_n,
            extrap_zone,
            inside_solid,
            denom,
            n_r_positive,
            n_r_negative,
            n_z_positive,
            n_z_negative,
        )

        self._restore_eta(eta, bounded_eta)

    # =================================================
    # =============== Helper Functions ================
    # =================================================
    def _find_bounding_box(self, inside):
        r_axis = np.any(inside, axis=1)
        z_axis = np.any(inside, axis=0)
        self.r_start, self.r_end = np.where(r_axis)[0][[0, -1]]
        self.z_start, self.z_end = np.where(z_axis)[0][[0, -1]]
        self.r_end += 1
        self.z_end += 1

        if self.r_start >= 2:
            self.r_start -= 2
            self.r_start_at_boundary = False

        if self.r_end <= self.grid_size_r - 2:
            self.r_end += 2
            self.r_end_at_boundary = False

        if self.z_start >= 2:
            self.z_start -= 2
            self.z_start_at_boundary = False

        if self.z_end <= self.grid_size_z - 2:
            self.z_end += 2
            self.z_end_at_boudary = False

        if min(self.r_end - self.r_start, self.z_end - self.z_start) < 6:
            raise ValueError("Too few grid points. Using higher resolution!")

    def _zones_setup(self, phi):
        inside_solid = phi[2:-2, 2:-2] <= -self.offset
        extrap_zone = (phi[2:-2, 2:-2] > -self.offset) & (
            phi[2:-2, 2:-2] < self.extrap_band
        )
        return inside_solid, extrap_zone

    def _compute_normal_upwind(self, phi):
        n_r = (
            np.maximum(
                3.0 * phi[2:-2, 2:-2] - 4.0 * phi[1:-3, 2:-2] + phi[:-4, 2:-2], 0
            )
            + np.minimum(
                -3.0 * phi[2:-2, 2:-2] + 4.0 * phi[3:-1, 2:-2] - phi[4:, 2:-2], 0
            )
        ) / (2 * self.dx)

        n_z = (
            np.maximum(
                3.0 * phi[2:-2, 2:-2] - 4.0 * phi[2:-2, 1:-3] + phi[2:-2, :-4], 0
            )
            + np.minimum(
                -3.0 * phi[2:-2, 2:-2] + 4.0 * phi[2:-2, 3:-1] - phi[2:-2, 4:], 0
            )
        ) / (2 * self.dx)

        n_mag = np.sqrt(n_r**2 + n_z**2)
        n_r /= n_mag + self.eps
        n_z /= n_mag + self.eps
        return n_r, n_z

    # normal * grad(soln_field) = rhs_field
    # soln_field and rhs_field must be of the same dimensions
    def _jacobi_iterate(
        self,
        soln_field,
        rhs_field,
        extrap_zone,
        inside_solid,
        denominator,
        n_r_positive,
        n_r_negative,
        n_z_positive,
        n_z_negative,
    ):
        residual = 1 + self.extrap_tol
        original = soln_field[2:-2, 2:-2].copy() * inside_solid
        previous = soln_field[2:-2, 2:-2].copy()

        while residual > self.extrap_tol:
            extrap_num_r = n_r_positive * (
                4.0 * soln_field[1:-3, 2:-2] - soln_field[:-4, 2:-2]
            ) - n_r_negative * (4.0 * soln_field[3:-1, 2:-2] - soln_field[4:, 2:-2])

            extrap_num_z = n_z_positive * (
                4.0 * soln_field[2:-2, 1:-3] - soln_field[2:-2, :-4]
            ) - n_z_negative * (4.0 * soln_field[2:-2, 3:-1] - soln_field[2:-2, 4:])

            soln_field[2:-2, 2:-2] = (
                original
                + extrap_zone
                * (2 * self.dx * rhs_field[2:-2, 2:-2] + extrap_num_r + extrap_num_z)
                / denominator
            )
            residual = la.norm(soln_field[2:-2, 2:-2] - previous)

            previous[...] = soln_field[2:-2, 2:-2]

    def _restore_eta(self, eta, bounded_eta):
        r_start_corr = 0 if self.r_start_at_boundary else 2
        r_end_corr = None if self.r_end_at_boundary else -2
        z_start_corr = 0 if self.z_start_at_boundary else 2
        z_end_corr = None if self.z_end_at_boudary else -2

        eta[...] *= 0

        eta[
            self.r_start
            + r_start_corr : self.r_end
            + int(0 if r_end_corr is None else r_end_corr),
            self.z_start
            + z_start_corr : self.z_end
            + int(0 if z_end_corr is None else z_end_corr),
        ] = bounded_eta[r_start_corr:r_end_corr, z_start_corr:z_end_corr]
