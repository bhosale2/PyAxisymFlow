#pragma once

#include "interpolation/particles_to_mesh_1D.hpp"
#include "interpolation/particles_to_mesh_2D.hpp"
#include "particle_kernels.hpp"

// particle to mesh interpolation using MP4 kernel in a 1D periodic domain
template <class Float>
inline void particles_to_mesh_1D_mp4(
    const Float particle_positions[], const int particle_positions_size,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size, Float output_field[],
    const int output_field_size, const Float delta_x) {
  detail::particles_to_mesh_with_offset_impl<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      particle_positions, particle_positions_size,
      input_field_at_particle_positions, input_field_at_particle_positions_size,
      output_field, output_field_size, delta_x);
};

// particle to mesh interpolation using Linear kernel in a
// 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_linear_kernel(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::LinearKernel>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using YangSmoothThreePointKernel kernel in a
// 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_yang_smooth_three_point_kernel(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::YangSmoothThreePointKernel>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using MP4 kernel in a 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_mp4(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using MP6 kernel in a 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_mp6(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::MP6>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using YangSmoothThreePointKernel kernel in a
// 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_unbounded_yang_smooth_three_point_kernel(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::YangSmoothThreePointKernel>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using Linear kernel in a
// 2D periodic domain
template <class Float>
inline void particles_to_mesh_2D_unbounded_linear_kernel(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::LinearKernel>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using MP4 kernel in a 2D unbounded domain
template <class Float>
inline void particles_to_mesh_2D_unbounded_mp4(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};

// particle to mesh interpolation using MP6 kernel in a 2D unbounded domain
template <class Float>
inline void particles_to_mesh_2D_unbounded_mp6(
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float input_field_at_particle_positions[],
    const int input_field_at_particle_positions_size0,
    const int input_field_at_particle_positions_size1,
    Float output_field_at_mesh[], const int output_field_at_mesh_size0,
    const int output_field_at_mesh_size1, const Float delta_x,
    const Float delta_y) {
  detail::particles_to_mesh_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::MP6>>(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, particle_positions_y,
      particle_positions_y_size0, particle_positions_y_size1,
      input_field_at_particle_positions,
      input_field_at_particle_positions_size0,
      input_field_at_particle_positions_size1, output_field_at_mesh,
      output_field_at_mesh_size0, output_field_at_mesh_size1, delta_x, delta_y);
};
