#pragma once
// For minmax
#include <algorithm>
// #include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "weno_2D_kernel.hpp"

#define CFS_RESTRICT __restrict__

// bindthem doesn't suppoer const pointers to objects (it interprets them as
// pointers to const objects)
// In this case, just wrap the function and use it

namespace detail {
  // TODO : Caveats : periodic, only in X : tag it
  template <class Float>
  void del_x_center_to_edge_periodic_impl(
      const Float* const CFS_RESTRICT qty_at_center,
      const int qty_at_center_size0, const int qty_at_center_size1,
      Float* const CFS_RESTRICT dqty_dx_at_edges,
      const int dqty_dx_at_edges_size0, const int dqty_dx_at_edges_size1,
      const Float x_factor) {
    const int stride = qty_at_center_size1;
    __attribute__((unused)) const int size =
        qty_at_center_size0 * qty_at_center_size1;
    for (int j = 0; j < qty_at_center_size0; ++j) {
      // Peel off

      // // Do first column only
      const int curr_index_zero = j * stride;
      dqty_dx_at_edges[curr_index_zero] =
          (qty_at_center[curr_index_zero] -
           qty_at_center[curr_index_zero + stride - 1]) *
          x_factor * 2.0;

      // Do one sided, include i = 0
      for (int i = 1; i < qty_at_center_size1; ++i) {
        const int curr_index = i + j * stride;

        dqty_dx_at_edges[curr_index] =
            (qty_at_center[curr_index] - qty_at_center[curr_index - 1]) *
            x_factor * 2.0;  //
      }

      // const int end_x(qty_at_center_size1 - 1);
      // // // Do last column only, wrap around
      // const int curr_index_end = end_x + j * stride;
      // dqty_dx_at_edges[curr_index_end] =
      //     (qty_at_center[j * stride] - qty_at_center[curr_index_end]) *
      //     x_factor * 2.0;
    }
  }

  template <class Float>
  void del_y_center_to_edge_periodic_impl(
      const Float* const CFS_RESTRICT qty_at_center,
      const int qty_at_center_size0, const int qty_at_center_size1,
      Float* const CFS_RESTRICT dqty_dy_at_edges,
      const int dqty_dy_at_edges_size0, const int dqty_dy_at_edges_size1,
      const Float y_factor) {
    const int stride = qty_at_center_size1;
    __attribute__((unused)) const int size =
        qty_at_center_size0 * qty_at_center_size1;

    // Do zeroth row only
    const int zero_row_offset(size - stride);
    for (int i = 0; i < qty_at_center_size1; ++i) {
      dqty_dy_at_edges[i] =
          (qty_at_center[i] - qty_at_center[i + zero_row_offset]) * y_factor *
          2.0;
    }

    // Exclude first row here, include last
    for (int j = 1; j < qty_at_center_size0; ++j) {
      for (int i = 0; i < stride; ++i) {
        const int curr_index_end = i + j * stride;
        dqty_dy_at_edges[curr_index_end] =
            (qty_at_center[curr_index_end] -
             qty_at_center[curr_index_end - stride]) *
            y_factor * 2.0;
      }
    }

    // const int end_y(qty_at_center_size0 - 1);
    // // Do j = qty_at_center_size0 - 1  finally
    // const int y_contrib_at_end(end_y * stride);
    // for (int i = 0; i < stride; ++i) {
    //   const int curr_index_end = i + y_contrib_at_end;
    //   dqty_dy_at_edges[curr_index_end] =
    //       (qty_at_center[i] - qty_at_center[curr_index_end]) * y_factor
    //       * 2.0;
    // }
  }

  template <class Float>
  void compute_godunov_ENO(
      const Float* const CFS_RESTRICT phi, const int phi_size0,
      const int phi_size1, const Float* const CFS_RESTRICT sign_function,
      const int sign_function_size0, const int sign_function_size1,
      Float* const CFS_RESTRICT dphi_dx, const int dphi_dx_size0,
      const int dphi_dx_size1, Float* const CFS_RESTRICT dphi_dy,
      const int dphi_dy_size0, const int dphi_dy_size1,
      Float* const CFS_RESTRICT result_buffer, const int result_buffer_size0,
      const int result_buffer_size1,
      Float* const CFS_RESTRICT left_cell_boundary_flux,
      const int left_cell_boundary_flux_size0,
      const int left_cell_boundary_flux_size1,
      Float* const CFS_RESTRICT right_cell_boundary_flux,
      const int right_cell_boundary_flux_size0,
      const int right_cell_boundary_flux_size1,
      Float* const CFS_RESTRICT bottom_cell_boundary_flux,
      const int bottom_cell_boundary_flux_size0,
      const int bottom_cell_boundary_flux_size1,
      Float* const CFS_RESTRICT top_cell_boundary_flux,
      const int top_cell_boundary_flux_size0,
      const int top_cell_boundary_flux_size1,
      const Float x_factor,  // 1/ dx
      const Float y_factor   // 1/ dy
  ) {
    // // Generate intermediate buffers for filling in derivatives
    // const int size = phi_size0 * phi_size1;
    // // TODO : Can this be statically cached?
    // // 0. Allocate memory on heap for computations
    // Float* dphi_dx(new Float[size]);
    // Float* dphi_dy(new Float[size]);

    // 1. Compute the one sided derivatives on the edges from cell centers
    detail::del_x_center_to_edge_periodic_impl(
        phi, phi_size0, phi_size1, dphi_dx, phi_size0, phi_size1, x_factor);
    detail::del_y_center_to_edge_periodic_impl(
        phi, phi_size0, phi_size1, dphi_dy, phi_size0, phi_size1, y_factor);

    // Populate the fluxes at the left, right, bottom and top edges
    // Note that the result buffer is used, but the values can be thrown away
    weno5_FD_2D_all_novec_reverse_iteration(
        phi, phi_size0, phi_size1, dphi_dx, phi_size0, phi_size1, dphi_dy,
        phi_size0, phi_size1, result_buffer, result_buffer_size0,
        result_buffer_size1, left_cell_boundary_flux,
        left_cell_boundary_flux_size0, left_cell_boundary_flux_size1,
        right_cell_boundary_flux, right_cell_boundary_flux_size0,
        right_cell_boundary_flux_size1, bottom_cell_boundary_flux,
        bottom_cell_boundary_flux_size0, bottom_cell_boundary_flux_size1,
        top_cell_boundary_flux, top_cell_boundary_flux_size0,
        top_cell_boundary_flux_size1,
        Float(0.0),  // x_alpha, for splitting
        Float(0.0),  // y_alpha, for splitting
        x_factor, y_factor);
    const int stride = phi_size1;

    constexpr std::size_t MIN = 0UL;
    constexpr std::size_t MAX = 1UL;

    // using MinMaxType = std::pair<Float, Float>;
    // std::array<MinMaxType, 4UL> reduction_buffer;

    // Reuse center fluxes as a result buffer
    for (int j = 0; j < phi_size0; ++j) {
      for (int i = 0; i < phi_size1; ++i) {
        const int curr_index = i + j * stride;
        // Why unnecessary copy into init list?
        // TODO : Can use a stack allocated array of pairs for doing the
        // reduction

        const auto left_minmax =
            // reduction_buffer[0] =
            std::minmax({left_cell_boundary_flux[curr_index], Float(0.0)});

        const auto right_minmax =
            // reduction_buffer[1] =
            std::minmax({right_cell_boundary_flux[curr_index], Float(0.0)});

        const auto bottom_minmax =
            // reduction_buffer[2] =
            std::minmax({bottom_cell_boundary_flux[curr_index], Float(0.0)});

        const auto top_minmax =
            // reduction_buffer[3] =
            std::minmax({top_cell_boundary_flux[curr_index], Float(0.0)});

        //       Float max_term(0.0), min_term(0.0);
        // #pragma unroll(4)
        //       for (int k = 0; k < 4; ++k) {
        //         const Float maxi(std::get<MAX>(reduction_buffer[k]));
        //         const Float mini(std::get<MIN>(reduction_buffer[k]));
        //         max_term += (maxi * maxi);
        //         min_term += (mini * mini);
        //       }

        const Float max_term = std::sqrt(
            (std::get<MAX>(left_minmax) * std::get<MAX>(left_minmax)) +
            (std::get<MIN>(right_minmax) * std::get<MIN>(right_minmax)) +
            (std::get<MAX>(bottom_minmax) * std::get<MAX>(bottom_minmax)) +
            (std::get<MIN>(top_minmax) * std::get<MIN>(top_minmax)));

        const Float min_term = std::sqrt(
            (std::get<MIN>(left_minmax) * std::get<MIN>(left_minmax)) +
            (std::get<MAX>(right_minmax) * std::get<MAX>(right_minmax)) +
            (std::get<MIN>(bottom_minmax) * std::get<MIN>(bottom_minmax)) +
            (std::get<MAX>(top_minmax) * std::get<MAX>(top_minmax)));

        // clang-format off
      result_buffer[curr_index] =
          std::max(sign_function[curr_index], Float(0.0)) *
              (max_term - Float(1.0)) +
          std::min(sign_function[curr_index], Float(0.0)) *
              // According to the python version, does a multiply with sign alter on
              // (Float(1.0) - min_term);
			  // According to Gibou(2017)?
			  (min_term - Float(1.0));
        // clang-format on
      }
    }

    // // N. Free memory because we are good boys
    // delete[] dphi_dx;
    // delete[] dphi_dy;
  }
}  // namespace detail

/*

  Caveat: There seems to be some difference between cpp and python's
  implementation of reinitialization. This is not surprising given the
  modifications (max/min) inside the reinitialization loop. This should
  be investigated sometime in the future.

 */

template <class Float>
inline void reinitialize_level_set_using_hamilton_jacobi(
    Float* /*restrict*/ phi_old, const int phi_old_size0,
    const int phi_old_size1, Float* /*restrict*/ phi_new,
    const int phi_new_size0, const int phi_new_size1,
    Float* /*const restrict*/ sign_function, const int sign_function_size0,
    const int sign_function_size1, Float* /*const restrict*/ dphi_dx,
    const int dphi_dx_size0, const int dphi_dx_size1,
    Float* /*const restrict*/ dphi_dy, const int dphi_dy_size0,
    const int dphi_dy_size1, Float* /*const restrict*/ result_buffer,
    const int result_buffer_size0, const int result_buffer_size1,
    Float* /*const restrict*/ left_cell_boundary_flux,
    const int left_cell_boundary_flux_size0,
    const int left_cell_boundary_flux_size1,
    Float* /*const restrict*/ right_cell_boundary_flux,
    const int right_cell_boundary_flux_size0,
    const int right_cell_boundary_flux_size1,
    Float* /*const restrict*/ bottom_cell_boundary_flux,
    const int bottom_cell_boundary_flux_size0,
    const int bottom_cell_boundary_flux_size1,
    Float* /*const restrict */ top_cell_boundary_flux,
    const int top_cell_boundary_flux_size0,
    const int top_cell_boundary_flux_size1, const Float delta_x,
    const Float delta_y,
    const Float x_factor,  // 1/ dx
    const Float y_factor,  // 1/ dy
    const Float reinit_band, const Float reinit_tolerance) {
  const Float delta_t(0.5 * delta_x);
  const std::size_t problem_size(phi_new_size0 * phi_new_size1);
  std::transform(phi_old, phi_old + problem_size, sign_function,
                 [delta_x_2 = delta_x * delta_x](Float phi) -> Float {
                   return phi / std::sqrt(phi * phi + delta_x_2);
                 });

  // Same
  // std::copy(phi_old, phi_old + problem_size, phi_new);
  std::memcpy(phi_new, phi_old, problem_size * sizeof(Float));

  auto reinit_error(std::numeric_limits<Float>::max());
  std::size_t n_iter(0UL);
  while (reinit_error > reinit_tolerance) {
    ++n_iter;
    // 1. Perform 1st stage SSPRK3
    // Update result_buffer flux, uses phi_old because its swapped at the last
    // stage wih phi_new
    detail::compute_godunov_ENO(
        phi_old, phi_old_size0, phi_old_size1, sign_function,
        sign_function_size0, sign_function_size1, dphi_dx, dphi_dx_size0,
        dphi_dx_size1, dphi_dy, dphi_dy_size0, dphi_dy_size1, result_buffer,
        result_buffer_size0, result_buffer_size1, left_cell_boundary_flux,
        left_cell_boundary_flux_size0, left_cell_boundary_flux_size1,
        right_cell_boundary_flux, right_cell_boundary_flux_size0,
        right_cell_boundary_flux_size1, bottom_cell_boundary_flux,
        bottom_cell_boundary_flux_size0, bottom_cell_boundary_flux_size1,
        top_cell_boundary_flux, top_cell_boundary_flux_size0,
        top_cell_boundary_flux_size1, x_factor, y_factor);
    for (std::size_t i = 0UL; i < problem_size; ++i)
      phi_new[i] = phi_old[i] - delta_t * result_buffer[i];

    // 2. Perform 2nd stage SSPRK3
    detail::compute_godunov_ENO(
        phi_new, phi_new_size0, phi_new_size1, sign_function,
        sign_function_size0, sign_function_size1, dphi_dx, dphi_dx_size0,
        dphi_dx_size1, dphi_dy, dphi_dy_size0, dphi_dy_size1, result_buffer,
        result_buffer_size0, result_buffer_size1, left_cell_boundary_flux,
        left_cell_boundary_flux_size0, left_cell_boundary_flux_size1,
        right_cell_boundary_flux, right_cell_boundary_flux_size0,
        right_cell_boundary_flux_size1, bottom_cell_boundary_flux,
        bottom_cell_boundary_flux_size0, bottom_cell_boundary_flux_size1,
        top_cell_boundary_flux, top_cell_boundary_flux_size0,
        top_cell_boundary_flux_size1, x_factor, y_factor);
    for (std::size_t i = 0UL; i < problem_size; ++i)
      phi_new[i] =
          0.75 * phi_old[i] + 0.25 * (phi_new[i] - delta_t * result_buffer[i]);

    // 3. Perform 3rd stage SSPRK3
    detail::compute_godunov_ENO(
        phi_new, phi_new_size0, phi_new_size1, sign_function,
        sign_function_size0, sign_function_size1, dphi_dx, dphi_dx_size0,
        dphi_dx_size1, dphi_dy, dphi_dy_size0, dphi_dy_size1, result_buffer,
        result_buffer_size0, result_buffer_size1, left_cell_boundary_flux,
        left_cell_boundary_flux_size0, left_cell_boundary_flux_size1,
        right_cell_boundary_flux, right_cell_boundary_flux_size0,
        right_cell_boundary_flux_size1, bottom_cell_boundary_flux,
        bottom_cell_boundary_flux_size0, bottom_cell_boundary_flux_size1,
        top_cell_boundary_flux, top_cell_boundary_flux_size0,
        top_cell_boundary_flux_size1, x_factor, y_factor);
    for (std::size_t i = 0UL; i < problem_size; ++i)
      phi_new[i] = Float(0.3333333333333333333333333333) * phi_old[i] +
                   Float(0.6666666666666666666666666667) *
                       (phi_new[i] - delta_t * result_buffer[i]);

    // 4. Check for degree of convergence in zone
    Float sum_numerator(0.0);
    std::size_t sumH(0UL);
    for (std::size_t i = 0UL; i < problem_size; ++i) {
      if (std::abs(phi_old[i]) < reinit_band) {
        sumH++;
        sum_numerator += std::abs(phi_new[i] - phi_old[i]);
      }
    }
    reinit_error = sum_numerator / (static_cast<Float>(sumH) + 1e-6);

    // 5. Don't copy old buffer, just switch pointers here
    // std::copy(phi_new, phi_new + problem_size, phi_old);
    // std::memcpy(phi_old, phi_new, problem_size * sizeof(Float));

    std::swap(phi_old, phi_new);

    // if (n_iter == 100)
    //   break;
  }

  // If n_iters is odd, then switch back pointers, else leave them alone
  // You can also use n_iter & 1 to check last digit, but this is more clearer
  // Python seems to have a pointer to the data anyway, so phi_new should always
  // give the updated value
  if (n_iter % 2)
    std::swap(phi_old, phi_new);

  // printf("Number of iterations %ld\n", n_iter);
  // printf("Convergence critertia %f\n", reinit_error);
}

template <class Float>
inline void reinitialize_level_set_using_hamilton_jacobi_heap_memory(
    Float* /*restrict*/ phi_old, const int phi_old_size0,
    const int phi_old_size1, Float* /*restrict*/ phi_new,
    const int phi_new_size0, const int phi_new_size1, const Float delta_x,
    const Float delta_y,
    const Float x_factor,  // 1/ dx
    const Float y_factor,  // 1/ dy
    const Float reinit_band, const Float reinit_tolerance) {
  const std::size_t problem_size = phi_old_size0 * phi_old_size1;

  // 0. Allocate memory on heap for computations
  // TODO : Can this be statically cached?
  // DONE : The memory can definitely be pooled into one allocation
  constexpr std::size_t n_buffers(8UL);
  const std::size_t allocation_size(n_buffers * problem_size);
  Float* heap_buffer(new Float[allocation_size]);
  // Can use memset but whatever
  std::fill(heap_buffer, heap_buffer + allocation_size, Float(0.0));
  Float* sign_buffer(heap_buffer);
  Float* dphi_dx(heap_buffer + problem_size);
  Float* dphi_dy(heap_buffer + 2 * problem_size);
  Float* flux_buffer(heap_buffer + 3 * problem_size);
  Float* left_derivative_buffer(heap_buffer + 4 * problem_size);
  Float* right_derivative_buffer(heap_buffer + 5 * problem_size);
  Float* bottom_derivative_buffer(heap_buffer + 6 * problem_size);
  Float* top_derivative_buffer(heap_buffer + 7 * problem_size);
  // Float* sign_buffer(new Float[problem_size]);
  // Float* dphi_dx(new Float[problem_size]);
  // Float* dphi_dy(new Float[problem_size]);
  // Float* flux_buffer(new Float[problem_size]);
  // Float* left_derivative_buffer(new Float[problem_size]);
  // Float* right_derivative_buffer(new Float[problem_size]);
  // Float* bottom_derivative_buffer(new Float[problem_size]);
  // Float* top_derivative_buffer(new Float[problem_size]);

  // 1. Process
  reinitialize_level_set_using_hamilton_jacobi(
      phi_old, phi_old_size0, phi_old_size1, phi_new, phi_new_size0,
      phi_new_size1, sign_buffer, phi_old_size0, phi_old_size1, dphi_dx,
      phi_old_size0, phi_old_size1, dphi_dy, phi_old_size0, phi_old_size1,
      flux_buffer, phi_old_size0, phi_old_size1, left_derivative_buffer,
      phi_old_size0, phi_old_size1, right_derivative_buffer, phi_old_size0,
      phi_old_size1, bottom_derivative_buffer, phi_old_size0, phi_old_size1,
      top_derivative_buffer, phi_old_size0, phi_old_size1, delta_x, delta_y,
      x_factor, y_factor, reinit_band, reinit_tolerance);

  // // N. Free memory because we are good boys
  delete[] heap_buffer;

  // // N. Free memory because we are good boys
  // delete[] sign_buffer;
  // delete[] dphi_dx;
  // delete[] dphi_dy;
  // delete[] flux_buffer;
  // delete[] left_derivative_buffer;
  // delete[] right_derivative_buffer;
  // delete[] bottom_derivative_buffer;
  // delete[] top_derivative_buffer;
}

#undef CFS_RESTRICT
