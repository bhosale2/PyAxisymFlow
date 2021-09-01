import numpy as np
from set_sim_params import dx, grid_size_z, grid_size_r
import scipy.sparse as spp
import scipy.sparse.linalg as sppla


def stokes_phi_init(R):
    """
    solves the Stokes potential function pseudo Poisson:
    d^2 phi / dr^2 + d^2 phi / dx^2 + d phi / dr / r
    = divg
    """
    R_diag = spp.diags(
        (R ** -1).reshape(
            -1,
        ),
        format="csc",
    )
    FDM_zz = spp.diags(
        [1, -2, 1], [-1, 0, 1], shape=(grid_size_z, grid_size_z), format="csc"
    ) / (dx ** 2)
    FDM_rr = spp.diags(
        [1, -2, 1], [-1, 0, 1], shape=(grid_size_r, grid_size_r), format="csc"
    ) / (dx ** 2)

    # neumann along Z
    FDM_zz[0, 1] *= 2.0
    FDM_zz[-1, -2] *= 2.0

    # nuemann along R
    FDM_rr[0, 1] *= 2.0
    FDM_rr[-1, -2] *= 2.0

    FDM_r = spp.diags(
        [-1, 1], [-1, 1], shape=(grid_size_r, grid_size_r), format="csc"
    ) / (2 * dx)

    # nuemann along R
    FDM_r[0, 1] = 0.0
    FDM_r[-1, -2] = 0.0

    Id_z = spp.identity(grid_size_z)

    Id1 = spp.diags(np.repeat(1.0, grid_size_r), format="csc")
    Id1[-1, -1] = 0.0
    mask_Id = spp.kron(Id1, Id_z)
    Id2 = spp.diags(np.repeat(0.0, grid_size_r), format="csc")
    Id2[-1, -1] = 1.0
    unmask_Id = spp.kron(Id2, Id_z)
    Id3 = spp.diags(np.repeat(1.0, grid_size_r), format="csc")
    Id3[0, 0] = 0.0
    FDM_final = mask_Id * (
        spp.kron(FDM_rr, Id_z) + spp.kron(Id3, FDM_zz) + R_diag * spp.kron(FDM_r, Id_z)
    ) + unmask_Id / (dx ** 2)

    M2 = sppla.spilu(FDM_final)
    M = sppla.LinearOperator(
        (grid_size_z * grid_size_r, grid_size_z * grid_size_r), M2.solve
    )
    LU_decomp = sppla.splu(FDM_final)

    return FDM_final, M, LU_decomp


def stokes_phi_solve_gmres(phi, FDM_matrix, precond_matrix, vel_divg):
    phi = phi.reshape(
        -1,
    )
    phi[...], _ = sppla.gmres(
        FDM_matrix,
        vel_divg.reshape(
            -1,
        ),
        x0=phi.copy(),
        M=precond_matrix,
    )
    phi = phi.reshape(grid_size_r, grid_size_z)


def stokes_phi_solve_LU(phi, LU_decomp, vel_divg):
    phi = phi.reshape(
        -1,
    )
    phi[...] = LU_decomp.solve(
        vel_divg.reshape(
            -1,
        )
    )
    phi = phi.reshape(grid_size_r, grid_size_z)
