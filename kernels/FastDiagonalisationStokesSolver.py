import numpy as np
import numpy.linalg as la
import scipy.sparse as spp


class FastDiagonalisationStokesSolver:
    def __init__(
            self,
            grid_size_r,
            grid_size_z,
            dx,
            real_dtype=np.float64,
            bc_type="homogenous_neumann_along_z_and_r",
    ):
        self.dx = dx
        self.grid_size_r = grid_size_r
        self.grid_size_z = grid_size_z
        self.real_dtype = real_dtype
        self.bc_type = bc_type
        self.radial_coord = np.linspace(dx / 2,
                                        grid_size_r * dx - dx / 2,
                                        grid_size_r).astype(real_dtype).reshape(grid_size_r, 1)

        poisson_matrix_z, poisson_matrix_r, derivative_matrix_r = self.construct_fdm_matrices()
        self.apply_boundary_conds_to_poisson_matrices(
            poisson_matrix_z, poisson_matrix_r, derivative_matrix_r
        )
        self.compute_spectral_decomp_of_poisson_matrices(
            poisson_matrix_z, poisson_matrix_r, derivative_matrix_r
        )

        # allocate buffer for spectral field manipulation
        self.spectral_field_buffer = np.zeros_like(self.inv_eig_val_matrix)

    def construct_fdm_matrices(self):
        """
        Construct the finite difference matrices
        """
        inv_dx2 = self.real_dtype(1 / self.dx / self.dx)
        inv_2dx = self.real_dtype(1 / 2 / self.dx)
        poisson_matrix_z = inv_dx2 * spp.diags(
            [-1, 2, -1],
            [-1, 0, 1],
            shape=(self.grid_size_z, self.grid_size_z),
            format="csr",
        )
        poisson_matrix_z = poisson_matrix_z.toarray().astype(self.real_dtype)
        poisson_matrix_r = inv_dx2 * spp.diags(
            [-1, 2, -1],
            [-1, 0, 1],
            shape=(self.grid_size_r, self.grid_size_r),
            format="csr",
        )
        poisson_matrix_r = poisson_matrix_r.toarray().astype(self.real_dtype)
        derivative_matrix_r = inv_2dx * spp.diags(
            [1, -1], [-1, 1],
            shape=(self.grid_size_r, self.grid_size_r),
            format="csr")
        derivative_matrix_r = derivative_matrix_r.toarray().astype(self.real_dtype)
        derivative_matrix_r[...] = derivative_matrix_r / self.radial_coord

        return poisson_matrix_z, poisson_matrix_r, derivative_matrix_r

    def apply_boundary_conds_to_poisson_matrices(
            self, poisson_matrix_z, poisson_matrix_r, derivative_matrix_r,
    ):
        """
        Apply boundary conditions to matrices
        """
        inv_dx2 = self.real_dtype(1 / self.dx / self.dx)
        if self.bc_type == "homogenous_neumann_along_z_and_r":
            # neumann at z=0 and r/z=L, but the modification below operates on
            # nodes at z=dx/2 and r/z=L-dx/2, because of the grid shift in sims.
            poisson_matrix_z[0, 0] = inv_dx2
            poisson_matrix_z[-1, -1] = inv_dx2
            poisson_matrix_r[-1, -1] = inv_dx2
            # neumann at R_max
            derivative_matrix_r[-1, -2] = 0

        elif self.bc_type == "homogenous_neumann_along_r_and_periodic_along_z":           
            poisson_matrix_z[0, -1] = poisson_matrix_z[0, 1]
            poisson_matrix_z[-1, 0] = poisson_matrix_z[-1, -2]
            poisson_matrix_r[-1, -1] = inv_dx2
            # neumann at R_max
            derivative_matrix_r[-1, -2] = 0

    def compute_spectral_decomp_of_poisson_matrices(
            self, poisson_matrix_z, poisson_matrix_r, derivative_matrix_r,
    ):
        """
        Compute spectral decomposition (eigenvalue and vectors) of the
        Poisson matrices
        """
        eig_vals_r, eig_vecs_r = la.eig(poisson_matrix_r - derivative_matrix_r)
        # sort eigenvalues in decreasing order
        idx = eig_vals_r.argsort()[::-1]
        eig_vals_r[...] = eig_vals_r[idx]
        eig_vecs_r[...] = eig_vecs_r[:, idx]
        self.eig_vecs_r = eig_vecs_r
        self.inv_of_eig_vecs_r = la.inv(eig_vecs_r)

        eig_vals_z, eig_vecs_z = la.eig(poisson_matrix_z)
        # sort eigenvalues in decreasing order
        idx = eig_vals_z.argsort()[::-1]
        eig_vals_z[...] = eig_vals_z[idx]
        eig_vecs_z[...] = eig_vecs_z[:, idx]
        self.tranpose_of_eig_vecs_z = np.transpose(eig_vecs_z)
        self.tranpose_of_inv_of_eig_vecs_z = np.transpose(la.inv(eig_vecs_z))

        eig_val_matrix = np.tile(
            eig_vals_z.reshape(1, self.grid_size_z), reps=(self.grid_size_r, 1)
        ) + np.tile(eig_vals_r.reshape(self.grid_size_r, 1), reps=(1, self.grid_size_z))
        self.inv_eig_val_matrix = self.real_dtype(1) / eig_val_matrix

    def solve(self, solution_field, rhs_field):
        """
        solves the Stokes stream function pseudo Poisson:
        d^2 solution_field / dr^2 + d^2 solution_field / dx^2
        - d solution_field / dr / r = -rhs_field
        """
        # transform to spectral space ("forward transform")
        la.multi_dot(
            [self.inv_of_eig_vecs_r, (rhs_field * self.radial_coord), self.tranpose_of_inv_of_eig_vecs_z],
            out=self.spectral_field_buffer,
        )

        # convolution (elementwise) in spectral space
        np.multiply(
            self.spectral_field_buffer,
            self.inv_eig_val_matrix,
            out=self.spectral_field_buffer,
        )

        # transform to physical space ("backward transform")
        la.multi_dot(
            [self.eig_vecs_r, self.spectral_field_buffer, self.tranpose_of_eig_vecs_z],
            out=solution_field,
        )
