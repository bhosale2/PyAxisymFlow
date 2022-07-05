import numpy as np
from set_sim_params import dx, grid_size_z, grid_size_r
import scipy.sparse as spp
import scipy.sparse.linalg as sppla


def stokes_psi_init(R):
    """
    solves the Stokes stream function pseudo Poisson:
    d^2 psi / dr^2 + d^2 psi / dx^2 - d psi / dr / r
    = -omega *  r
    """
    R_diag = spp.diags(
        (R**-1).reshape(
            -1,
        ),
        format="csc",
    )
    FDM_zz = spp.diags(
        [1, -2, 1], [-1, 0, 1], shape=(grid_size_z, grid_size_z), format="csc"
    ) / (dx**2)
    FDM_rr = spp.diags(
        [1, -2, 1], [-1, 0, 1], shape=(grid_size_r, grid_size_r), format="csc"
    ) / (dx**2)

    # neumann along Z
    FDM_zz[0, 1] *= 2.0
    FDM_zz[-1, -2] *= 2.0

    # nuemann at Rmax
    FDM_rr[-1, -2] *= 2.0

    FDM_r = spp.diags(
        [-1, 1], [-1, 1], shape=(grid_size_r, grid_size_r), format="csc"
    ) / (2 * dx)
    # nuemann at Rmax
    FDM_r[-1, -2] = 0.0

    # higher order 1 side deriviative try, doesnt work
    # FDM_zz[0, 0] = 1.0 / (dx ** 2)
    # FDM_zz[0, 1] = -2.0 / (dx ** 2)
    # FDM_zz[0, 2] = 1.0 / (dx ** 2)
    # FDM_zz[-1, -1] = 1.0 / (dx ** 2)
    # FDM_zz[-1, -2] = -2.0 / (dx ** 2)
    # FDM_zz[-1, -3] = 1.0 / (dx ** 2)
    # FDM_rr[-1, -1] = 1.0 / (dx ** 2)
    # FDM_rr[-1, -2] = -2.0 / (dx ** 2)
    # FDM_rr[-1, -3] = 1.0 / (dx ** 2)
    # FDM_r[-1, -1] = 1.0 / dx
    # FDM_r[-1, -2] = -1.0 / dx

    Id_z = spp.identity(grid_size_z)

    Id1 = spp.diags(np.repeat(1.0, grid_size_r), format="csc")
    Id1[0, 0] = 0.0
    mask_Id = spp.kron(Id1, Id_z)
    Id2 = spp.diags(np.repeat(0.0, grid_size_r), format="csc")
    Id2[0, 0] = 1.0
    unmask_Id = spp.kron(Id2, Id_z)
    Id3 = spp.diags(np.repeat(1.0, grid_size_r), format="csc")
    Id3[-1, -1] = 0.0
    FDM_final = mask_Id * (
        spp.kron(FDM_rr, Id_z) + spp.kron(Id3, FDM_zz) - R_diag * spp.kron(FDM_r, Id_z)
    ) + unmask_Id / (dx**2)

    M2 = sppla.spilu(FDM_final)
    M = sppla.LinearOperator(
        (grid_size_z * grid_size_r, grid_size_z * grid_size_r), M2.solve
    )
    LU_decomp = sppla.splu(FDM_final)

    return FDM_final, M, LU_decomp


def stokes_psi_solve_gmres(psi, FDM_matrix, precond_matrix, vorticity, R):
    psi = psi.reshape(
        -1,
    )
    psi[...], _ = sppla.gmres(
        FDM_matrix,
        (-R * vorticity).reshape(
            -1,
        ),
        x0=psi.copy(),
        M=precond_matrix,
    )
    psi = psi.reshape(grid_size_r, grid_size_z)


def stokes_psi_solve_LU(psi, LU_decomp, vorticity, R):
    psi = psi.reshape(
        -1,
    )
    psi[...] = LU_decomp.solve(
        (-R * vorticity).reshape(
            -1,
        )
    )
    psi = psi.reshape(grid_size_r, grid_size_z)
