"""Kernels for elementwise operations."""
import numpy as np

import pystencils as ps

import sympy as sp


def gen_elementwise_sum_pyst_kernel(
    real_t=np.float64,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """elementwise sum kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = "float32" if real_t == np.float32 else "float64"
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype, default_number_float=pyst_dtype, cpu_openmp=num_threads
    )

    if field_type == "scalar":
        grid_info = (
            f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
        )

        @ps.kernel
        def _elementwise_sum_stencil():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            sum_field[0, 0] @= field_1[0, 0] + field_2[0, 0]

    elif field_type == "vector":
        grid_info = (
            f"2, {fixed_grid_size[0]}, {fixed_grid_size[1]}"
            if fixed_grid_size
            else "3D"
        )

        @ps.kernel
        def _elementwise_sum_stencil():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            sum_field[0, 0, 0] @= field_1[0, 0, 0] + field_2[0, 0, 0]

    elementwise_sum_pyst_kernel = ps.create_kernel(
        _elementwise_sum_stencil, config=kernel_config
    ).compile()
    return elementwise_sum_pyst_kernel


def gen_set_fixed_val_pyst_kernel(
    real_t=np.float64,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """2D set field to fixed value kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = "float32" if real_t == np.float32 else "float64"
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype, default_number_float=pyst_dtype, cpu_openmp=num_threads
    )
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _set_fixed_val_stencil():
        field = ps.fields(f"field : {pyst_dtype}[{grid_info}]")
        fixed_val = sp.symbols("fixed_val")
        field[0, 0] @= fixed_val

    set_fixed_val_pyst_kernel = ps.create_kernel(
        _set_fixed_val_stencil, config=kernel_config
    ).compile()
    if field_type == "scalar":
        return set_fixed_val_pyst_kernel
    elif field_type == "vector":

        def vector_field_set_fixed_val_pyst_kernel(
            vector_field,
            fixed_vals,
        ):
            """Set vector field to fixed value.

            Sets spatially constant values for a vector field,
            assumes shape of fields (2, n, n).
            """
            set_fixed_val_pyst_kernel(
                field=vector_field[0],
                fixed_val=fixed_vals[0],
            )
            set_fixed_val_pyst_kernel(
                field=vector_field[1],
                fixed_val=fixed_vals[1],
            )

        return vector_field_set_fixed_val_pyst_kernel
