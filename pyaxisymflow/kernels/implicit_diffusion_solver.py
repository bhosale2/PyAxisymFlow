import numpy as np
import numpy.linalg as la
import scipy.sparse as spp


class ImplicitEulerDiffusionStepper:
    def __init__(
        self,
        time_step,
        kinematic_viscosity,
        grid_size_r,
        grid_size_z,
        dx,
        real_dtype=np.float64,
    ):
        self.time_step = time_step
        self.nu_times_dt = self.time_step * kinematic_viscosity
        self.dx = dx
        self.grid_size_r = grid_size_r
        self.grid_size_z = grid_size_z
        self.real_dtype = real_dtype
        self.radial_coord = (
            np.linspace(dx / 2, grid_size_r * dx - dx / 2, grid_size_r)
            .astype(real_dtype)
            .reshape(grid_size_r, 1)
        )

        (
            poisson_matrix_z,
            poisson_matrix_r,
            non_poisson_matrix_r,
        ) = self.construct_fdm_matrices()
        self.compute_spectral_decomp_of_poisson_matrices(
            poisson_matrix_z, poisson_matrix_r, non_poisson_matrix_r
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
            [1, -2, 1],
            [-1, 0, 1],
            shape=(self.grid_size_z, self.grid_size_z),
            format="csr",
        )
        poisson_matrix_z = poisson_matrix_z.toarray().astype(self.real_dtype)
        poisson_matrix_r = inv_dx2 * spp.diags(
            [1, -2, 1],
            [-1, 0, 1],
            shape=(self.grid_size_r, self.grid_size_r),
            format="csr",
        )
        poisson_matrix_r = poisson_matrix_r.toarray().astype(self.real_dtype)
        # These are the extra terms other the classical laplacian in the
        # diffusion operator
        non_poisson_matrix_r = inv_2dx * spp.diags(
            [-1, 1], [-1, 1], shape=(self.grid_size_r, self.grid_size_r), format="csr"
        )
        non_poisson_matrix_r = non_poisson_matrix_r.toarray().astype(self.real_dtype)
        non_poisson_matrix_r[
            ...
        ] = non_poisson_matrix_r / self.radial_coord - np.identity(self.grid_size_r) / (
            self.radial_coord**2
        )

        return poisson_matrix_z, poisson_matrix_r, non_poisson_matrix_r

    def compute_spectral_decomp_of_poisson_matrices(
        self,
        poisson_matrix_z,
        poisson_matrix_r,
        non_poisson_matrix_r,
    ):
        """
        Compute spectral decomposition (eigenvalue and vectors) of the
        Poisson matrices
        """
        eig_vals_r, eig_vecs_r = la.eig(poisson_matrix_r + non_poisson_matrix_r)
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

        eig_val_matrix = self.real_dtype(1.0) - self.nu_times_dt * (
            np.tile(eig_vals_z.reshape(1, self.grid_size_z), reps=(self.grid_size_r, 1))
            + np.tile(
                eig_vals_r.reshape(self.grid_size_r, 1), reps=(1, self.grid_size_z)
            )
        )
        self.inv_eig_val_matrix = self.real_dtype(1) / eig_val_matrix

    def step(self, vorticity_field, dt):
        """
        Performs an implicit Euler timestep for diffusion in
        axisym coordinates.
        """
        # check if the simulations uses the same timestep as
        # the one used for assembling the matrices
        if dt != self.time_step:
            raise ValueError(
                "dt should be constant throughout the simulation if using "
                "implicit diffusion! Please use the value of dt being used "
                "to initialize the implicit diffusion stepper"
            )

        # transform to spectral space ("forward transform")
        la.multi_dot(
            [
                self.inv_of_eig_vecs_r,
                vorticity_field,
                self.tranpose_of_inv_of_eig_vecs_z,
            ],
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
            out=vorticity_field,
        )
