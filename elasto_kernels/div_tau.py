from numba import njit


# @njit(fastmath=True)
@njit(parallel=True)
# @njit(parallel=True, fastmath=True)
def update_vorticity_from_solid_stress(
    vorticity, tau_z, tau_r, tau11, tau12, tau22, R, dt, dx
):
    tau_z[1:-1, 1:-1] = (
        tau11[1:-1, 2:] - tau11[1:-1, :-2] + tau12[2:, 1:-1] - tau12[:-2, 1:-1]
    ) / (2 * dx) + tau12[1:-1, 1:-1] / R[1:-1, 1:-1]
    tau_z[0, 1:-1] = (
        tau11[0, 2:]
        - tau11[0, :-2]
        - tau12[2, 1:-1]
        + 4 * tau12[1, 1:-1]
        - 3 * tau12[0, 1:-1]
    ) / (2 * dx) + tau12[0, 1:-1] / R[0, 1:-1]

    tau_r[1:-1, 1:-1] = (
        tau12[1:-1, 2:] - tau12[1:-1, :-2] + tau22[2:, 1:-1] - tau22[:-2, 1:-1]
    ) / (2 * dx) + tau22[1:-1, 1:-1] / R[1:-1, 1:-1]
    tau_r[0, 1:-1] = (
        tau12[0, 2:]
        - tau12[0, :-2]
        - tau22[2, 1:-1]
        + 4 * tau22[1, 1:-1]
        - 3 * tau22[0, 1:-1]
    ) / (2 * dx) + tau22[0, 1:-1] / R[0, 1:-1]

    vorticity[1:-1, 1:-1] += (
        dt
        * (tau_r[1:-1, 2:] - tau_r[1:-1, :-2] - tau_z[2:, 1:-1] + tau_z[:-2, 1:-1])
        / (2 * dx)
    )
