from numba import njit


# @njit(fastmath=True)
@njit(fastmath=True, parallel=True)
def diffusion_RK2_unb(vorticity, temp_vorticity, R, nu, dt, dx):
    """
    update field due to diffusion using a RK2 time stepper
    """
    temp_vorticity[...] = vorticity
    temp_vorticity[1:-1, 1:-1] += (
        0.5
        * nu
        * dt
        * (
            (
                vorticity[2:, 1:-1]
                + vorticity[:-2, 1:-1]
                + vorticity[1:-1, 2:]
                + vorticity[1:-1, :-2]
                - 4 * vorticity[1:-1, 1:-1]
            )
            / (dx ** 2)
            + (vorticity[2:, 1:-1] - vorticity[:-2, 1:-1]) / (2 * dx) / R[1:-1, 1:-1]
            - vorticity[1:-1, 1:-1] * (R[1:-1, 1:-1] ** -2)
        )
    )
    vorticity[1:-1, 1:-1] += (
        nu
        * dt
        * (
            (
                temp_vorticity[2:, 1:-1]
                + temp_vorticity[:-2, 1:-1]
                + temp_vorticity[1:-1, 2:]
                + temp_vorticity[1:-1, :-2]
                - 4 * temp_vorticity[1:-1, 1:-1]
            )
            / (dx ** 2)
            + (temp_vorticity[2:, 1:-1] - temp_vorticity[:-2, 1:-1])
            / (2 * dx)
            / R[1:-1, 1:-1]
            - temp_vorticity[1:-1, 1:-1] * (R[1:-1, 1:-1] ** -2)
        )
    )


# @njit(parallel=True)
def diffusion_RK2_unb_diffrho(vorticity, temp_vorticity, R, density, nu, dt, dx):
    """
    update vorticity field due to diffusion using a RK2 time stepper for variable density
    """
    temp_vorticity[...] = vorticity
    temp_vorticity[1:-1, 1:-1] += (
        0.5
        * nu
        * dt
        * (
            (
                vorticity[2:, 1:-1]
                + vorticity[:-2, 1:-1]
                + vorticity[1:-1, 2:]
                + vorticity[1:-1, :-2]
                - 4 * vorticity[1:-1, 1:-1]
            )
            / (dx ** 2)
            + (vorticity[2:, 1:-1] - vorticity[:-2, 1:-1]) / (2 * dx) / R[1:-1, 1:-1]
            - vorticity[1:-1, 1:-1] * (R[1:-1, 1:-1] ** -2)
        )
        / density[1:-1, 1:-1]
    )
    vorticity[1:-1, 1:-1] += (
        nu
        * dt
        * (
            (
                temp_vorticity[2:, 1:-1]
                + temp_vorticity[:-2, 1:-1]
                + temp_vorticity[1:-1, 2:]
                + temp_vorticity[1:-1, :-2]
                - 4 * temp_vorticity[1:-1, 1:-1]
            )
            / (dx ** 2)
            + (temp_vorticity[2:, 1:-1] - temp_vorticity[:-2, 1:-1])
            / (2 * dx)
            / R[1:-1, 1:-1]
            - temp_vorticity[1:-1, 1:-1] * (R[1:-1, 1:-1] ** -2)
        )
        / density[1:-1, 1:-1]
    )
