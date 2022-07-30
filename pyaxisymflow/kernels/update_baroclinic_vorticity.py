from numba import njit


@njit(cache=True)
def update_baroclinic_vorticity(vorticity, u_z, u_r, old_u_z, old_u_r, density, dt, dx):
    """
    performs baroclinic update term to the vorticity
    """
    # Du_dt = du_dt + u * (u * gradient_T) + v * (gradient * u)
    # Dv_dt = dv_dt + u * (v * gradient_T) + v * (gradient * v)
    # rho_x = rho * gradient_T
    # rho_y = gradient * rho
    # baroc_term = (Du_dt - bx) * rho_y - (Dv_dt - by) * rho_x
    # baroc_term /= rho
    Du_z_dt = (
        (u_z[1:-1, 1:-1] - old_u_z[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_z[1:-1, 2:] - u_z[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2 * dx)
    )
    Du_r_dt = (
        (u_r[1:-1, 1:-1] - old_u_r[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_r[2:, 1:-1] - u_r[:-2, 1:-1]) / (2 * dx)
    )
    vorticity[1:-1, 1:-1] += (
        dt
        * (
            Du_z_dt * (density[2:, 1:-1] - density[:-2, 1:-1]) / (2 * dx)
            - Du_r_dt * (density[1:-1, 2:] - density[1:-1, :-2]) / (2 * dx)
        )
        / density[1:-1, 1:-1]
    )


@njit(cache=True)
def update_baroclinic_vorticity_penal(
    vorticity, u_z, u_r, old_u_z, old_u_r, density, penal_term_z, penal_term_r, dt, dx
):
    """
    performs baroclinic update term to the vorticity
    includes the penalisation term
    """
    Du_z_dt = (
        (u_z[1:-1, 1:-1] - old_u_z[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_z[1:-1, 2:] - u_z[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2 * dx)
    )
    Du_r_dt = (
        (u_r[1:-1, 1:-1] - old_u_r[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_r[2:, 1:-1] - u_r[:-2, 1:-1]) / (2 * dx)
    )
    vorticity[1:-1, 1:-1] += (
        dt
        * (
            (Du_z_dt - penal_term_z[1:-1, 1:-1])
            * (density[2:, 1:-1] - density[:-2, 1:-1])
            / (2 * dx)
            - (Du_r_dt - penal_term_r[1:-1, 1:-1])
            * (density[1:-1, 2:] - density[1:-1, :-2])
            / (2 * dx)
        )
        / density[1:-1, 1:-1]
    )


@njit(cache=True)
def update_baroclinic_vorticity_diff_penal(
    vorticity,
    u_z,
    u_r,
    old_u_z,
    old_u_r,
    density,
    penal_term_z,
    penal_term_r,
    R,
    nu,
    dt,
    dx,
):
    """
    performs baroclinic update term to the vorticity
    includes diffusion and penalisation term
    """
    Du_z_dt = (
        (u_z[1:-1, 1:-1] - old_u_z[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_z[1:-1, 2:] - u_z[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2 * dx)
    )
    Du_r_dt = (
        (u_r[1:-1, 1:-1] - old_u_r[1:-1, 1:-1]) / dt
        + u_z[1:-1, 1:-1] * (u_r[1:-1, 2:] - u_r[1:-1, :-2]) / (2 * dx)
        + u_r[1:-1, 1:-1] * (u_r[2:, 1:-1] - u_r[:-2, 1:-1]) / (2 * dx)
    )
    del2_u_z = (
        u_z[2:, 1:-1]
        + u_z[:-2, 1:-1]
        + u_z[1:-1, 2:]
        + u_z[1:-1, :-2]
        - 4 * u_z[1:-1, 1:-1]
    ) / (dx**2) + (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2 * dx) / R[1:-1, 1:-1]
    del2_u_r = (
        (
            u_r[2:, 1:-1]
            + u_r[:-2, 1:-1]
            + u_r[1:-1, 2:]
            + u_r[1:-1, :-2]
            - 4 * u_r[1:-1, 1:-1]
        )
        / (dx**2)
        + (u_r[2:, 1:-1] - u_r[:-2, 1:-1]) / (2 * dx) / R[1:-1, 1:-1]
        - u_r[1:-1, 1:-1] * (R[1:-1, 1:-1] ** -2)
    )

    vorticity[1:-1, 1:-1] += (
        dt
        * (
            (Du_z_dt - penal_term_z[1:-1, 1:-1] - nu * del2_u_z)
            * (density[2:, 1:-1] - density[:-2, 1:-1])
            / (2 * dx)
            - (Du_r_dt - penal_term_r[1:-1, 1:-1] - nu * del2_u_r)
            * (density[1:-1, 2:] - density[1:-1, :-2])
            / (2 * dx)
        )
        / density[1:-1, 1:-1]
    )
