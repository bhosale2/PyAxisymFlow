#pragma once

#include "interpolation/mesh_to_particles_1D.hpp"
#include "interpolation/mesh_to_particles_2D.hpp"
#include "interpolation/wrap_particles.hpp"
#include "particle_kernels.hpp"

// Wrap particles around in a domain
// Have the option of doing it
// 1. in a halo region of 10 points from start and end
// 2. all particles
// To not cause undefined behvaior , the limits do not exceed the size,
// and so whichever is minimal between the strategy and size is chosen.
template <class Float>
inline void wrap_particles_around_1D_domain(Float particle_positions[],
                                            const int particle_positions_size,
                                            const Float domain_start,
                                            const Float domain_end) {
  detail::wrap_particles_around_1D_domain_impl<
      detail::WrappingStrategy::FirstN>(
      particle_positions, particle_positions_size, domain_start, domain_end);
}

// mesh to particle interpolation using MP4 kernel in a 1D periodic domain
template <class Float>
inline void mesh_to_particles_1D_mp4(
    const Float input_field[], const int input_field_size,
    const Float particle_positions[], const int particle_positions_size,
    Float output_field[], const int output_field_size, const Float delta_x) {
  detail::mesh_to_particles_with_offset_impl<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      input_field, input_field_size, particle_positions,
      particle_positions_size, output_field, output_field_size, delta_x);
};

// Wraps particles around a two dimensional domain
// Delegates calls to 1D version : please look at its documentation
// to see options
template <class Float>
inline void wrap_particles_around_2D_domain(
    Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    const Float domain_start_x, const Float domain_end_x,
    const Float domain_start_y, const Float domain_end_y) {
  detail::wrap_particles_around_2D_domain_in_x_impl(
      particle_positions_x, particle_positions_x_size0,
      particle_positions_x_size1, domain_start_x, domain_end_x);
  detail::wrap_particles_around_2D_domain_in_y_impl(
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, domain_start_y, domain_end_y);
}

/*
 * Instantations of different M2P routines : bindthem relies on text matching
 hence macros cannot be used
 */

// mesh to particle interpolation using LinearKernel in a 2D
// periodic domain
template <class Float>
inline void mesh_to_particles_2D_linear_kernel(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::LinearKernel>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using YangSmoothThreePointKernel in a 2D
// periodic domain
template <class Float>
inline void mesh_to_particles_2D_yang_smooth_three_point_kernel(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::YangSmoothThreePointKernel>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using MP4 kernel in a 2D periodic domain
template <class Float>
inline void mesh_to_particles_2D_mp4(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using MP6 kernel in a 2D periodic domain
template <class Float>
inline void mesh_to_particles_2D_mp6(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D<
      Float, kernels::KernelWrapper<kernels::MP6>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using LinearKernel in a 2D
// unbounded domain
template <class Float>
inline void mesh_to_particles_2D_unbounded_linear_kernel(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::LinearKernel>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using YangSmoothThreePointKernel in a 2D
// unbounded domain
template <class Float>
inline void mesh_to_particles_2D_unbounded_yang_smooth_three_point_kernel(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::YangSmoothThreePointKernel>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using MP4 kernel in a 2D unbounded domain
template <class Float>
inline void mesh_to_particles_2D_unbounded_mp4(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::MP4>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};

// mesh to particle interpolation using MP6 kernel in a 2D unbounded domain
template <class Float>
inline void mesh_to_particles_2D_unbounded_mp6(
    const Float input_field_x[], const int input_field_x_size0,
    const int input_field_x_size1, const Float input_field_y[],
    const int input_field_y_size0, const int input_field_y_size1,
    const Float particle_positions_x[], const int particle_positions_x_size0,
    const int particle_positions_x_size1, const Float particle_positions_y[],
    const int particle_positions_y_size0, const int particle_positions_y_size1,
    Float output_field_x[], const int output_field_x_size0,
    const int output_field_x_size1, Float output_field_y[],
    const int output_field_y_size0, const int output_field_y_size1,
    const Float delta_x, const Float delta_y) {
  detail::mesh_to_particles_with_offset_impl_2D_unbounded<
      Float, kernels::KernelWrapper<kernels::MP6>>(
      input_field_x, input_field_x_size0, input_field_x_size1, input_field_y,
      input_field_y_size0, input_field_y_size1, particle_positions_x,
      particle_positions_x_size0, particle_positions_x_size1,
      particle_positions_y, particle_positions_y_size0,
      particle_positions_y_size1, output_field_x, output_field_x_size0,
      output_field_x_size1, output_field_y, output_field_y_size0,
      output_field_y_size1, delta_x, delta_y);
};
