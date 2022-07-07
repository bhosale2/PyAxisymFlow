"""Kernels for performing advection timestep."""
import numpy as np

from pyst_kernels.advection_flux import (
    gen_advection_flux_conservative_eno3_pyst_kernel,
)
from pyst_kernels.elementwise_ops import (
    gen_elementwise_sum_pyst_kernel,
    gen_set_fixed_val_pyst_kernel,
)


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel(
        real_t=np.float64, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """Advection (ENO3 stencil) Euler forward timestep generator."""
    elementwise_sum_pyst_kernel = gen_elementwise_sum_pyst_kernel(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    set_fixed_val_pyst_kernel = gen_set_fixed_val_pyst_kernel(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    advection_flux_conservative_eno3_pyst_kernel = (
        gen_advection_flux_conservative_eno3_pyst_kernel(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_kernel(
            field, advection_flux, velocity, dt_by_dx
    ):
        """Advection (ENO3 stencil) Euler forward timestep.

        Performs an inplace advection timestep (using ENO3 stencil)
        using Euler forward, for a 2D field (n, n).
        """
        set_fixed_val_pyst_kernel(field=advection_flux, fixed_val=0)
        advection_flux_conservative_eno3_pyst_kernel(
            advection_flux=advection_flux,
            field=field,
            velocity=velocity,
            inv_dx=-dt_by_dx,
        )
        elementwise_sum_pyst_kernel(
            sum_field=field, field_1=field, field_2=advection_flux
        )

    return advection_timestep_euler_forward_conservative_eno3_pyst_kernel

