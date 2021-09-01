#pragma once
#include <iostream>
#include <memory>

#include "weno_kernel.hpp"
#include "weno_kernel_constants.h"

/*
  Performs 2D WENO, but without flux splitting.
  As a result it only works for fields always upwind (such as c_x = X, c_y = Y)
  and fails for more complicated fields.
 */
template <class Float>
void weno5_FD_2D_novec(
    const Float uin[], const int uin_size0, const int uin_size1,
    const Float sampled_flux_at_nodes_x[],
    const int sampled_flux_at_nodes_x_size0,
    const int sampled_flux_at_nodes_x_size1,
    const Float sampled_flux_at_nodes_y[],
    const int sampled_flux_at_nodes_y_size0,
    const int sampled_flux_at_nodes_y_size1, Float total_flux_at_center[],
    const int total_flux_at_center_size0, const int total_flux_at_center_size1,
    Float boundary_flux[], const int boundary_flux_size0,
    const int boundary_flux_size1, const Float x_alpha, const Float y_alpha,
    const Float x_factor,  // 1/ dx
    const Float y_factor   // 1/ dy
) {
  // This should do the x-periodic
  for (int i = 0; i < sampled_flux_at_nodes_x_size1; ++i) {
    int curr_index = i * sampled_flux_at_nodes_x_size0;
    weno5_FD_1D_novec<Float>(
        &uin[curr_index], uin_size0, &sampled_flux_at_nodes_x[curr_index],
        sampled_flux_at_nodes_x_size0, &total_flux_at_center[curr_index],
        total_flux_at_center_size0, &boundary_flux[curr_index],
        boundary_flux_size0, x_alpha, x_factor);
  }

  // y-periodic, change the way data is accessed
  // or transpose all data
  const int stride = sampled_flux_at_nodes_y_size0;
  const int size =
      sampled_flux_at_nodes_y_size0 * sampled_flux_at_nodes_y_size1;
  for (int j = 0; j < sampled_flux_at_nodes_y_size1; ++j) {
    for (int i = 0; i < sampled_flux_at_nodes_y_size0; ++i) {
      // clang-format on

      // Reconstruct polynomials at (i + 1/2) location
      // $\hat{f}_{i+1/2}^{-} from f(u)$
      // clang-format off
	  const Float f_mm = sampled_flux_at_nodes_y[(i + (j - 2) * stride + size) % size];
	  const Float f_m  = sampled_flux_at_nodes_y[(i + (j - 1) * stride + size) % size];
	  const Float f    = sampled_flux_at_nodes_y[           i + j *stride            ];
	  const Float f_p  = sampled_flux_at_nodes_y[(i + (j + 1) * stride) % size];
	  const Float f_pp = sampled_flux_at_nodes_y[(i + (j + 2) * stride) % size];

      // clang-format on
#ifdef INTERNAL_DEBUG_
      printf("Index mm %d", (i + (j - 2) * stride + size) % size);
      printf("Index m %d", (i + (j - 1) * stride + size) % size);
      printf("Index p %d", (i + (j + 1) * stride) % size);
      printf("Index pp %d", (i + (j + 2) * stride) % size);
#endif

#ifdef INTERNAL_DEBUG_
      printf("Sampled mm at position %d is %f\n", i, f_mm);
      printf("Sampled m at position %d is %f\n", i, f_m);
      printf("Sampled p at position %d is %f\n", i, f_p);
      printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

      // clang-format off
	  const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	  const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	  const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
      // clang-format on

      // Smoothness indicators (\beta)
      const Float beta_1n =
          1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
              (f_mm - 2.0 * f_m + f) +
          0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
      const Float beta_2n = 1.08333333333333333333 * (f_m - 2.0 * f + f_p) *
                                (f_m - 2.0 * f + f_p) +
                            0.25 * (f_m - f_p) * (f_m - f_p);
      const Float beta_3n =
          1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
              (f_pp - 2.0 * f_p + f) +
          0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

      // Construct non-linear, non-normalized weights
      const Float w_tilde_1n =
          WENO_GAMMA_1 / ((WENO_EPSILON + beta_1n) * (WENO_EPSILON + beta_1n));
      const Float w_tilde_2n =
          WENO_GAMMA_2 / ((WENO_EPSILON + beta_2n) * (WENO_EPSILON + beta_2n));
      const Float w_tilde_3n =
          WENO_GAMMA_3 / ((WENO_EPSILON + beta_3n) * (WENO_EPSILON + beta_3n));
      const Float w_sum_n = w_tilde_1n + w_tilde_2n + w_tilde_3n;

      // Construct WENO weights now
      const Float w1n = w_tilde_1n / w_sum_n;
      const Float w2n = w_tilde_2n / w_sum_n;
      const Float w3n = w_tilde_3n / w_sum_n;

      // Reconstruct numerical flux at cell boundary $u_{i+1/2}^{-}$
      // flux_n
      // No += here because we are simply reusing the buffer
      // The total_flux already contains the x contribution
      boundary_flux[i + j * stride] =
          w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
    }
  }

  for (int j = 0; j < sampled_flux_at_nodes_x_size1; ++j) {
    for (int i = 0; i < sampled_flux_at_nodes_x_size0; ++i) {
      // clang-format on
      total_flux_at_center[i + j * stride] +=
          y_factor * (boundary_flux[i + j * stride] -
                      boundary_flux[(i + (j - 1) * stride + size) % size]);
    }
  }
};

/*
  Performs 2D WENO, with flux splitting.
  Iterates over the field in an incorrect order, arising from confusion between
  numpy's data layout and sizes.
*/
template <class Float>
void weno5_FD_2D_all_novec(
    const Float uin[], const int uin_size0, const int uin_size1,
    const Float sampled_flux_at_nodes_x[],
    const int sampled_flux_at_nodes_x_size0,
    const int sampled_flux_at_nodes_x_size1,
    const Float sampled_flux_at_nodes_y[],
    const int sampled_flux_at_nodes_y_size0,
    const int sampled_flux_at_nodes_y_size1, Float total_flux_at_center[],
    const int total_flux_at_center_size0, const int total_flux_at_center_size1,
    Float boundary_flux[], const int boundary_flux_size0,
    const int boundary_flux_size1, const Float x_alpha, const Float y_alpha,
    const Float x_factor,  // 1/ dx
    const Float y_factor   // 1/ dy
) {
  // This should do the x-periodic
  for (int i = 0; i < sampled_flux_at_nodes_x_size1; ++i) {
    int curr_index = i * sampled_flux_at_nodes_x_size0;
    weno5_FD_1D_all_novec_optimized_two<Float>(
        &uin[curr_index], uin_size0, &sampled_flux_at_nodes_x[curr_index],
        sampled_flux_at_nodes_x_size0, &total_flux_at_center[curr_index],
        total_flux_at_center_size0, &boundary_flux[curr_index],
        boundary_flux_size0, x_alpha, x_factor);
  }

  // y-periodic, change the way data is accessed
  // or transpose all data
  const int stride = sampled_flux_at_nodes_y_size0;
  const int size =
      sampled_flux_at_nodes_y_size0 * sampled_flux_at_nodes_y_size1;
  for (int j = 0; j < sampled_flux_at_nodes_y_size1; ++j) {
    for (int i = 0; i < sampled_flux_at_nodes_y_size0; ++i) {
      // clang-format on

      // Reconstruct polynomials at (i + 1/2) location
      // $\hat{f}_{i+1/2}^{-} from f(u)$

      int curr_index = (i + (j - 2) * stride + size) % size;
      const Float f_mm = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                y_alpha * uin[curr_index]);
      curr_index = (i + (j - 1) * stride + size) % size;
      const Float f_m = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                               y_alpha * uin[curr_index]);
      curr_index = i + j * stride;
      const Float f = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                             y_alpha * uin[curr_index]);
      curr_index = (i + (j + 1) * stride) % size;
      const Float f_p = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                               y_alpha * uin[curr_index]);
      curr_index = (i + (j + 2) * stride) % size;
      const Float f_pp = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                y_alpha * uin[curr_index]);

      // clang-format off
	  const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	  const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	  const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
      // clang-format on

      // Smoothness indicators (\beta)
      const Float beta_1n =
          1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
              (f_mm - 2.0 * f_m + f) +
          0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
      const Float beta_2n = 1.08333333333333333333 * (f_m - 2.0 * f + f_p) *
                                (f_m - 2.0 * f + f_p) +
                            0.25 * (f_m - f_p) * (f_m - f_p);
      const Float beta_3n =
          1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
              (f_pp - 2.0 * f_p + f) +
          0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

      // Construct non-linear, non-normalized weights
      const Float w_tilde_1n =
          WENO_GAMMA_1 / ((WENO_EPSILON + beta_1n) * (WENO_EPSILON + beta_1n));
      const Float w_tilde_2n =
          WENO_GAMMA_2 / ((WENO_EPSILON + beta_2n) * (WENO_EPSILON + beta_2n));
      const Float w_tilde_3n =
          WENO_GAMMA_3 / ((WENO_EPSILON + beta_3n) * (WENO_EPSILON + beta_3n));
      const Float w_sum_n = w_tilde_1n + w_tilde_2n + w_tilde_3n;

      // Construct WENO weights now
      const Float w1n = w_tilde_1n / w_sum_n;
      const Float w2n = w_tilde_2n / w_sum_n;
      const Float w3n = w_tilde_3n / w_sum_n;

      // Reconstruct numerical flux at cell boundary $u_{i+1/2}^{-}$
      // flux_n
      // No += here because we are simply reusing the buffer
      // The total_flux already contains the x contribution
      boundary_flux[i + j * stride] =
          w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
    }
  }

  // Do the negative flux
  for (int j = 0; j < sampled_flux_at_nodes_y_size1; ++j) {
    for (int i = 0; i < sampled_flux_at_nodes_y_size0; ++i) {
      ///
      int curr_index = (i + (j - 2) * stride + size) % size;
      const Float f_mm = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                y_alpha * uin[curr_index]);
      curr_index = (i + (j - 1) * stride + size) % size;
      const Float f_m = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                               y_alpha * uin[curr_index]);
      curr_index = i + j * stride;
      const Float f = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                             y_alpha * uin[curr_index]);
      curr_index = (i + (j + 1) * stride) % size;
      const Float f_p = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                               y_alpha * uin[curr_index]);
      curr_index = (i + (j + 2) * stride) % size;
      const Float f_pp = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                y_alpha * uin[curr_index]);
      // clang-format off
	  const Float fhat_1p = (0.3333333333333333334 * f  - 0.1666666666666666667 * f_mm  + 0.833333333333333333 * f_m  );
	  const Float fhat_2p = (0.3333333333333333334 * f_m   - 0.1666666666666666667 * f_p  + 0.833333333333333333 * f  );
	  const Float fhat_3p = (0.3333333333333333334 * f_pp  - 1.1666666666666666667 * f_p  + 1.833333333333333333 * f  );
      // clang-format on

      const Float beta_1p =
          1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
              (f_mm - 2.0 * f_m + f) +
          0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
      const Float beta_2p = 1.08333333333333333333 * (f_m - 2.0 * f + f_p) *
                                (f_m - 2.0 * f + f_p) +
                            0.25 * (f_m - f_p) * (f_m - f_p);
      const Float beta_3p =
          1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
              (f_pp - 2.0 * f_p + f) +
          0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

      const Float w_tilde_1p =
          WENO_GAMMA_3 / ((WENO_EPSILON + beta_1p) * (WENO_EPSILON + beta_1p));
      const Float w_tilde_2p =
          WENO_GAMMA_2 / ((WENO_EPSILON + beta_2p) * (WENO_EPSILON + beta_2p));
      const Float w_tilde_3p =
          WENO_GAMMA_1 / ((WENO_EPSILON + beta_3p) * (WENO_EPSILON + beta_3p));
      const Float w_sum_p = w_tilde_1p + w_tilde_2p + w_tilde_3p;

      const Float w1p = w_tilde_1p / w_sum_p;
      const Float w2p = w_tilde_2p / w_sum_p;
      const Float w3p = w_tilde_3p / w_sum_p;

      curr_index = (i + (j - 1) * stride + size) % size;
      boundary_flux[curr_index] +=
          w1p * fhat_1p + w2p * fhat_2p + w3p * fhat_3p;
    }
  }

  for (int j = 0; j < sampled_flux_at_nodes_x_size1; ++j) {
    for (int i = 0; i < sampled_flux_at_nodes_x_size0; ++i) {
      // clang-format on
      total_flux_at_center[i + j * stride] +=
          y_factor * (boundary_flux[i + j * stride] -
                      boundary_flux[(i + (j - 1) * stride + size) % size]);
    }
  }
};

/*
  Performs 2D WENO, with flux splitting.
  Iterates over the field in the correct order.
*/
template <class Float>
void weno5_FD_2D_all_novec_reverse_iteration(
    const Float* uin, const int uin_size0, const int uin_size1,
    const Float* sampled_flux_at_nodes_x,
    const int sampled_flux_at_nodes_x_size0,
    const int sampled_flux_at_nodes_x_size1,
    const Float* sampled_flux_at_nodes_y,
    const int sampled_flux_at_nodes_y_size0,
    const int sampled_flux_at_nodes_y_size1, Float* total_flux_at_center,
    const int total_flux_at_center_size0, const int total_flux_at_center_size1,
    Float* left_cell_boundary_flux, const int left_cell_boundary_flux_size0,
    const int left_cell_boundary_flux_size1, Float* right_cell_boundary_flux,
    const int right_cell_boundary_flux_size0,
    const int right_cell_boundary_flux_size1, Float* bottom_cell_boundary_flux,
    const int bottom_cell_boundary_flux_size0,
    const int bottom_cell_boundary_flux_size1, Float* top_cell_boundary_flux,
    const int top_cell_boundary_flux_size0,
    const int top_cell_boundary_flux_size1, const Float x_alpha,
    const Float y_alpha,
    const Float x_factor,  // 1/ dx
    const Float y_factor   // 1/ dy
) {
  // This should do the x-periodic
  for (int j = 0; j < sampled_flux_at_nodes_x_size0; ++j) {
    int curr_index = j * sampled_flux_at_nodes_x_size1;
    weno5_FD_1D_all_novec_optimized<Float>(
        &uin[curr_index], uin_size1, &sampled_flux_at_nodes_x[curr_index],
        sampled_flux_at_nodes_x_size1, &total_flux_at_center[curr_index],
        total_flux_at_center_size1, &left_cell_boundary_flux[curr_index],
        left_cell_boundary_flux_size1, &right_cell_boundary_flux[curr_index],
        right_cell_boundary_flux_size1, x_alpha, x_factor);
  }

  // y-periodic, change the way data is accessed
  // or transpose all data
  const int stride = sampled_flux_at_nodes_y_size1;
  const int size =
      sampled_flux_at_nodes_y_size0 * sampled_flux_at_nodes_y_size1;
  // Do the positive fluxes first
  {
    for (int j = 0; j < sampled_flux_at_nodes_y_size0; ++j) {
      for (int i = 0; i < sampled_flux_at_nodes_y_size1; ++i) {
        // clang-format on
        // Skip flux splitting for now

        // Reconstruct polynomials at (i + 1/2) location
        // $\hat{f}_{i+1/2}^{-} from f(u)$

        int curr_index = (i + (j - 2) * stride + size) % size;
        const Float f_mm = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                  y_alpha * uin[curr_index]);

        curr_index = (i + (j - 1) * stride + size) % size;
        const Float f_m = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                 y_alpha * uin[curr_index]);

        curr_index = i + j * stride;
        const Float f = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                               y_alpha * uin[curr_index]);

        curr_index = (i + (j + 1) * stride) % size;
        const Float f_p = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                 y_alpha * uin[curr_index]);

        curr_index = (i + (j + 2) * stride) % size;
        const Float f_pp = 0.5 * (sampled_flux_at_nodes_y[curr_index] +
                                  y_alpha * uin[curr_index]);

        const Float fhat_1n =
            (0.3333333333333333334 * f_mm - 1.1666666666666666667 * f_m +
             1.833333333333333333 * f);
        const Float fhat_2n =
            (0.3333333333333333334 * f_p - 0.1666666666666666667 * f_m +
             0.833333333333333333 * f);
        const Float fhat_3n =
            (0.3333333333333333334 * f - 0.1666666666666666667 * f_pp +
             0.833333333333333333 * f_p);

        // Smoothness indicators (\beta)
        const Float beta_1n =
            1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
                (f_mm - 2.0 * f_m + f) +
            0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
        const Float beta_2n = 1.08333333333333333333 * (f_m - 2.0 * f + f_p) *
                                  (f_m - 2.0 * f + f_p) +
                              0.25 * (f_m - f_p) * (f_m - f_p);
        const Float beta_3n =
            1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                (f_pp - 2.0 * f_p + f) +
            0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

        // Construct non-linear, non-normalized weights
        const Float w_tilde_1n = WENO_GAMMA_1 / ((WENO_EPSILON + beta_1n) *
                                                 (WENO_EPSILON + beta_1n));
        const Float w_tilde_2n = WENO_GAMMA_2 / ((WENO_EPSILON + beta_2n) *
                                                 (WENO_EPSILON + beta_2n));
        const Float w_tilde_3n = WENO_GAMMA_3 / ((WENO_EPSILON + beta_3n) *
                                                 (WENO_EPSILON + beta_3n));
        const Float w_sum_n = w_tilde_1n + w_tilde_2n + w_tilde_3n;

        // Construct WENO weights now
        const Float w1n = w_tilde_1n / w_sum_n;
        const Float w2n = w_tilde_2n / w_sum_n;
        const Float w3n = w_tilde_3n / w_sum_n;

        // Reconstruct numerical flux at cell boundary $u_{i}{j+1/2}^{-}$
        bottom_cell_boundary_flux[i + j * stride] =
            w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
      }
    }
  }

  // Do the negative fluxes next
  {
    for (int j = 0; j < sampled_flux_at_nodes_y_size0; ++j) {
      for (int i = 0; i < sampled_flux_at_nodes_y_size1; ++i) {
        ///
        int curr_index = (i + (j - 2) * stride + size) % size;
        const Float f_mm = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                  y_alpha * uin[curr_index]);

        curr_index = (i + (j - 1) * stride + size) % size;
        const Float f_m = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                 y_alpha * uin[curr_index]);

        curr_index = i + j * stride;
        const Float f = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                               y_alpha * uin[curr_index]);

        curr_index = (i + (j + 1) * stride) % size;
        const Float f_p = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                 y_alpha * uin[curr_index]);

        curr_index = (i + (j + 2) * stride) % size;
        const Float f_pp = 0.5 * (sampled_flux_at_nodes_y[curr_index] -
                                  y_alpha * uin[curr_index]);

        const Float fhat_1p =
            (0.3333333333333333334 * f - 0.1666666666666666667 * f_mm +
             0.833333333333333333 * f_m);
        const Float fhat_2p =
            (0.3333333333333333334 * f_m - 0.1666666666666666667 * f_p +
             0.833333333333333333 * f);
        const Float fhat_3p =
            (0.3333333333333333334 * f_pp - 1.1666666666666666667 * f_p +
             1.833333333333333333 * f);

        const Float beta_1p =
            1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
                (f_mm - 2.0 * f_m + f) +
            0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
        const Float beta_2p = 1.08333333333333333333 * (f_m - 2.0 * f + f_p) *
                                  (f_m - 2.0 * f + f_p) +
                              0.25 * (f_m - f_p) * (f_m - f_p);
        const Float beta_3p =
            1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                (f_pp - 2.0 * f_p + f) +
            0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

        const Float w_tilde_1p = WENO_GAMMA_3 / ((WENO_EPSILON + beta_1p) *
                                                 (WENO_EPSILON + beta_1p));
        const Float w_tilde_2p = WENO_GAMMA_2 / ((WENO_EPSILON + beta_2p) *
                                                 (WENO_EPSILON + beta_2p));
        const Float w_tilde_3p = WENO_GAMMA_1 / ((WENO_EPSILON + beta_3p) *
                                                 (WENO_EPSILON + beta_3p));
        const Float w_sum_p = w_tilde_1p + w_tilde_2p + w_tilde_3p;

        const Float w1p = w_tilde_1p / w_sum_p;
        const Float w2p = w_tilde_2p / w_sum_p;
        const Float w3p = w_tilde_3p / w_sum_p;

        curr_index = (i + (j - 1) * stride + size) % size;
        // Reconstruct numerical flux at cell boundary $u_{i}{j-1/2}^{+}$
        // Put in new buffer to support reinitialization
        top_cell_boundary_flux[curr_index] =
            w1p * fhat_1p + w2p * fhat_2p + w3p * fhat_3p;
      }
    }
  }

  // Finally compute center fluxes
  {
    // First do only the bottom row, i.e. j = 0
    const int offset = size - stride;
    for (int i = 0; i < sampled_flux_at_nodes_x_size1; ++i) {
      total_flux_at_center[i] +=
          y_factor * (bottom_cell_boundary_flux[i] + top_cell_boundary_flux[i] -
                      bottom_cell_boundary_flux[i + offset] -
                      top_cell_boundary_flux[i + offset]);
    }

    // Then peel off rest of the rows
    for (int j = 1; j < sampled_flux_at_nodes_x_size0; ++j) {
      for (int i = 0; i < sampled_flux_at_nodes_x_size1; ++i) {
        // clang-format on
        const int curr_index = i + j * stride;
        total_flux_at_center[curr_index] +=
            y_factor * (bottom_cell_boundary_flux[curr_index] +
                        top_cell_boundary_flux[curr_index] -
                        bottom_cell_boundary_flux[curr_index - stride] -
                        top_cell_boundary_flux[curr_index - stride]);
      }
    }
  }
};

/*
 This is a test function and should not be exported
 */
/*
template <class Float>
void test_two_dimensions_memory_layout(const Float uin[], const int uin_size0,
                                       const int uin_size1) {
  std::cout << "Size0" << uin_size0 << std::endl;
  std::cout << "Size1" << uin_size1 << std::endl;
  for (int j = 0; j < uin_size0; ++j) {
    for (int i = 0; i < uin_size1; ++i) {
      const int current_index = i + j * uin_size1;
      const Float curr_element = uin[current_index];
      auto pp = std::addressof(uin[current_index]);
      std::cout << "i : " << i << ", j : " << j << ", curr :" << current_index
                << ", addr :" << pp << ", val :" << curr_element << std::endl;
    }
  }
}
*/
