#pragma once

#include "weno_kernel_constants.h"

// Misleading as we aren't in a FV world
// Domain cells (I{i}) reference:
//
//                |           |   u(i)    |           |
//                |  u(i-1)   |___________|           |
//                |___________|           |   u(i+1)  |
//                |           |           |___________|
//             ...|-----0-----|-----0-----|-----0-----|...
//                |    i-1    |     i     |    i+1    |
//                |-         +|-         +|-         +|
//              i-3/2       i-1/2       i+1/2       i+3/2

// The logical (array) indexing is done in the following fashion:
//
//
//        NODES    ... 0------0------0------0------0------0 ...
//                    i-2    i-1     i     i+1    i+2    i+3
//
//      Left Flux  ... ---|------|------|------|------|--- ...
//                       i-2    i-1     i     i+1    i+2
//
//      Right Flux ... ---|------|------|------|------|--- ...
//                       i-2    i-1     i     i+1    i+2
//
// As a result for computing left flux (i) we need nodal values
// (i-2, i-1, i, i+1, i+2)
// For computing right flux (i) we need nodal values:
// (i-1, i, i+1, i+2, i+3)

template <class Float>
void weno5_FD_1D(const Float uin[], const int uin_size,
                 const Float sampled_flux_at_nodes[],
                 const int sampled_flux_at_nodes_size,
                 Float total_flux_at_center[],
                 const int total_flux_at_center_size, Float boundary_flux[],
                 const int boundary_flux_size, const Float alpha,
                 const Float factor,  // 1/ dx
                 const int start, const int end) {
  return;
};

template <class Float>
void weno5_FD_1D_novec(const Float uin[], const int uin_size,
                       const Float sampled_flux_at_nodes[],
                       const int sampled_flux_at_nodes_size,
                       Float total_flux_at_center[],
                       const int total_flux_at_center_size,
                       Float boundary_flux[], const int boundary_flux_size,
                       const Float alpha,
                       const Float factor  // 1/ dx
) {
  // Manually unroll index at 0, 1, 2
  // clang-format off
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    // Skip flux splitting for now
    // printf("Sampled flux at position %d is %f\n", i,
    // sampled_flux_at_nodes[i]);

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = sampled_flux_at_nodes[(i - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size];
	const Float f_m  = sampled_flux_at_nodes[(i - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size];
	const Float f    = sampled_flux_at_nodes[                  i                 ];
	const Float f_p  = sampled_flux_at_nodes[(i + 1) % sampled_flux_at_nodes_size];
	const Float f_pp = sampled_flux_at_nodes[(i + 2) % sampled_flux_at_nodes_size];
#ifdef INTERNAL_DEBUG_
	printf("Index mm %d", (i - 2) % sampled_flux_at_nodes_size);
	printf("Index m %d", (i - 1) % sampled_flux_at_nodes_size);
	printf("Index p %d", (i + 1) % sampled_flux_at_nodes_size);
	printf("Index pp %d", (i + 2) % sampled_flux_at_nodes_size);
#endif

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float beta_1n =
        1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
            (f_mm - 2.0 * f_m + f) +
        0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2n =
        1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p) +
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
    boundary_flux[i] = w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
  }
  // clang-format off
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    // printf("Boundary flux at position %d is %f\n", i, boundary_flux[i]);
    total_flux_at_center[i] =
        factor *
        (boundary_flux[i] - boundary_flux[(i - 1 + sampled_flux_at_nodes_size) %
                                          sampled_flux_at_nodes_size]);
  }
};

template <class Float>
void weno5_FD_1D_novec_optimized(
    const Float uin[], const int uin_size, const Float sampled_flux_at_nodes[],
    const int sampled_flux_at_nodes_size, Float total_flux_at_center[],
    const int total_flux_at_center_size, Float boundary_flux[],
    const int boundary_flux_size, const Float alpha,
    const Float factor  // 1/ dx
) {
  // Manually unroll index at 0, 1, 2
  // clang-format off
  int j = 0;
  Float f_m = sampled_flux_at_nodes[(j - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size];
  Float f = sampled_flux_at_nodes[(j - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size];
  Float f_p = sampled_flux_at_nodes[                  j                 ];
  Float f_pp = sampled_flux_at_nodes[(j + 1) % sampled_flux_at_nodes_size];

  // Actually (f_mm - 2.0 * f_m + f), but here replaced by (f_m - 2.0*f + f_p) for memory
  Float cold_factor =
	1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p);
  // Actually (f_m - 2.0*f + f_p), but here replaced by (f - 2.0*f_p + f_pp) for memory
  Float warm_factor =
	1.08333333333333333333 * (f - 2.0 * f_p + f_pp) * (f - 2.0 * f_p + f_pp);

  for (int i=0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    // Skip flux splitting for now
    // printf("Sampled flux at position %d is %f\n", i,
    // sampled_flux_at_nodes[i]);

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = f_m;
	f_m  = f;
	f    = f_p;
	f_p  = f_pp;
	f_pp = sampled_flux_at_nodes[(i + 2) % sampled_flux_at_nodes_size];
#ifdef INTERNAL_DEBUG_
	printf("Index mm %d", (i - 2) % sampled_flux_at_nodes_size);
	printf("Index m %d", (i - 1) % sampled_flux_at_nodes_size);
	printf("Index p %d", (i + 1) % sampled_flux_at_nodes_size);
	printf("Index pp %d", (i + 2) % sampled_flux_at_nodes_size);
#endif

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float hot_factor = 1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                             (f_pp - 2.0 * f_p + f);
    const Float beta_1n = cold_factor + 0.25 * (f_mm - 4.0 * f_m + 3.0 * f) *
                                            (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2n = warm_factor + 0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3n = hot_factor + 0.25 * (f_pp - 4.0 * f_p + 3.0 * f) *
                                           (f_pp - 4.0 * f_p + 3.0 * f);
    cold_factor = warm_factor;
    warm_factor = hot_factor;

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
    boundary_flux[i] = w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
  }
  // clang-format off
  total_flux_at_center[0] = factor * (boundary_flux[0] - boundary_flux[sampled_flux_at_nodes_size - 1]);
  for (int i = 1; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    // printf("Boundary flux at position %d is %f\n", i, boundary_flux[i]);
    total_flux_at_center[i] =
        factor * (boundary_flux[i] - boundary_flux[i - 1]);
  }
};

template <class Float>
void weno5_FD_1D_all_novec(const Float uin[], const int uin_size,
                           const Float sampled_flux_at_nodes[],
                           const int sampled_flux_at_nodes_size,
                           Float total_flux_at_center[],
                           const int total_flux_at_center_size,
                           Float boundary_flux[], const int boundary_flux_size,
                           const Float alpha,
                           const Float factor  // 1/ dx
) {
  // I can't see a straightforward way of not repeating code and getting
  // performance at the same time for a flux-splitting implementation yet. So we
  // repeat code and process the splitting.

  // Manually unroll index at 0, 1, 2
  // clang-format off
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
#ifdef INTERNAL_DEBUG_
    printf("Sampled flux at position %d is %f\n", i, sampled_flux_at_nodes[i]);
#endif

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = 0.5 * (sampled_flux_at_nodes[(i - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] + alpha * uin[(i - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size]);
	const Float f_m  = 0.5 * (sampled_flux_at_nodes[(i - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] + alpha * uin[(i - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] ) ;
	const Float f    = 0.5 * (sampled_flux_at_nodes[                  i                 ] + alpha * uin[i]);
	const Float f_p  = 0.5 * (sampled_flux_at_nodes[(i + 1) % sampled_flux_at_nodes_size] + alpha * uin[(i + 1) % sampled_flux_at_nodes_size]);
	const Float f_pp = 0.5 * (sampled_flux_at_nodes[(i + 2) % sampled_flux_at_nodes_size] + alpha * uin[(i + 2) % sampled_flux_at_nodes_size]);

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float beta_1n =
        1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
            (f_mm - 2.0 * f_m + f) +
        0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2n =
        1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p) +
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
    boundary_flux[i] = w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
  }

  // Note : here i is not the current cell, but one plus (for upwinding)
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i) {
    // clang-format on
#ifdef INTERNAL_DEBUG_
    printf("Sampled flux at position %d is %f\n", i, sampled_flux_at_nodes[i]);
#endif

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = 0.5 * (sampled_flux_at_nodes[(i - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] - alpha * uin[(i - 2 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size]);
	const Float f_m  = 0.5 * (sampled_flux_at_nodes[(i - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] - alpha * uin[(i - 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size] ) ;
	const Float f    = 0.5 * (sampled_flux_at_nodes[                  i                 ] - alpha * uin[i]);
	const Float f_p  = 0.5 * (sampled_flux_at_nodes[(i + 1) % sampled_flux_at_nodes_size] - alpha * uin[(i + 1 + sampled_flux_at_nodes_size) % sampled_flux_at_nodes_size]);
	const Float f_pp = 0.5 * (sampled_flux_at_nodes[(i + 2) % sampled_flux_at_nodes_size] - alpha * uin[(i + 2) % sampled_flux_at_nodes_size]);

#ifdef INTERNAL_DEBUG_
	printf("Index pp %d", (i + 2) % sampled_flux_at_nodes_size);
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1p = (0.3333333333333333334 * f  - 0.1666666666666666667 * f_mm  + 0.833333333333333333 * f_m  );
	const Float fhat_2p = (0.3333333333333333334 * f_m   - 0.1666666666666666667 * f_p  + 0.833333333333333333 * f  );
	const Float fhat_3p = (0.3333333333333333334 * f_pp  - 1.1666666666666666667 * f_p  + 1.833333333333333333 * f  );
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float beta_1p =
        1.08333333333333333333 * (f_mm - 2.0 * f_m + f) *
            (f_mm - 2.0 * f_m + f) +
        0.25 * (f_mm - 4.0 * f_m + 3.0 * f) * (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2p =
        1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p) +
        0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3p =
        1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
            (f_pp - 2.0 * f_p + f) +
        0.25 * (f_pp - 4.0 * f_p + 3.0 * f) * (f_pp - 4.0 * f_p + 3.0 * f);

    // Construct non-linear, non-normalized weights
    const Float w_tilde_1p =
        WENO_GAMMA_3 / ((WENO_EPSILON + beta_1p) * (WENO_EPSILON + beta_1p));
    const Float w_tilde_2p =
        WENO_GAMMA_2 / ((WENO_EPSILON + beta_2p) * (WENO_EPSILON + beta_2p));
    const Float w_tilde_3p =
        WENO_GAMMA_1 / ((WENO_EPSILON + beta_3p) * (WENO_EPSILON + beta_3p));
    const Float w_sum_p = w_tilde_1p + w_tilde_2p + w_tilde_3p;

    // Construct WENO weights now
    const Float w1p = w_tilde_1p / w_sum_p;
    const Float w2p = w_tilde_2p / w_sum_p;
    const Float w3p = w_tilde_3p / w_sum_p;

    // Reconstruct numerical flux at cell boundary $u_{i+1/2}^{-}$
    // flux_n
    boundary_flux[(i - 1 + sampled_flux_at_nodes_size) %
                  sampled_flux_at_nodes_size] +=
        w1p * fhat_1p + w2p * fhat_2p + w3p * fhat_3p;
  }

  // clang-format off
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
#ifdef INTERNAL_DEBUG_
    printf("Boundary flux at position %d is %f\n", i, boundary_flux[i]);
#endif
    total_flux_at_center[i] =
        factor *
        (boundary_flux[i] - boundary_flux[(i - 1 + sampled_flux_at_nodes_size) %
                                          sampled_flux_at_nodes_size]);
  }
};

template <class Float>
void weno5_FD_1D_all_novec_optimized(
    const Float uin[], const int uin_size, const Float sampled_flux_at_nodes[],
    const int sampled_flux_at_nodes_size, Float total_flux_at_center[],
    const int total_flux_at_center_size, Float left_cell_boundary_flux[],
    const int left_cell_boundary_flux_size, Float right_cell_boundary_flux[],
    const int right_cell_boundary_flux_size, const Float alpha,
    const Float factor  // 1/ dx
) {
  // Manually unroll index at 0, 1, 2
  // clang-format off

  Float f_m =  0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 2] + alpha * uin[sampled_flux_at_nodes_size - 2]);
  Float f = 0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 1] + alpha * uin[sampled_flux_at_nodes_size - 1] ) ;
  Float f_p = 0.5 * (sampled_flux_at_nodes[0] + alpha * uin[0]);
  Float f_pp = 0.5 * (sampled_flux_at_nodes[1] + alpha * uin[1]);

  // Actually (f_mm - 2.0 * f_m + f), but here replaced by (f_m - 2.0*f + f_p) for memory
  Float cold_factor =
	1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p);
  // Actually (f_m - 2.0*f + f_p), but here replaced by (f - 2.0*f_p + f_pp) for memory
  Float warm_factor =
	1.08333333333333333333 * (f - 2.0 * f_p + f_pp) * (f - 2.0 * f_p + f_pp);

  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = f_m;
	f_m  = f;
	f    = f_p;
	f_p  = f_pp;
	const int curr_index = (i + 2) % sampled_flux_at_nodes_size;
	f_pp = 0.5 * (sampled_flux_at_nodes[curr_index]  + alpha * uin[curr_index]);

#ifdef INTERNAL_DEBUG_
	printf("Index pp %d", (i + 2) % sampled_flux_at_nodes_size);
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float hot_factor = 1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                             (f_pp - 2.0 * f_p + f);
    const Float beta_1n = cold_factor + 0.25 * (f_mm - 4.0 * f_m + 3.0 * f) *
                                            (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2n = warm_factor + 0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3n = hot_factor + 0.25 * (f_pp - 4.0 * f_p + 3.0 * f) *
                                           (f_pp - 4.0 * f_p + 3.0 * f);
    cold_factor = warm_factor;
    warm_factor = hot_factor;

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
    // Put values in a new buffer to support reinitialization
    left_cell_boundary_flux[i] = w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
  }

  f_m = 0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 2] -
               alpha * uin[sampled_flux_at_nodes_size - 2]);
  f = 0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 1] -
             alpha * uin[sampled_flux_at_nodes_size - 1]);
  f_p = 0.5 * (sampled_flux_at_nodes[0] - alpha * uin[0]);
  f_pp = 0.5 * (sampled_flux_at_nodes[1] - alpha * uin[1]);

  // Actually (f_mm - 2.0 * f_m + f), but here replaced by (f_m - 2.0*f + f_p)
  // for memory
  cold_factor =
      1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p);
  // Actually (f_m - 2.0*f + f_p), but here replaced by (f - 2.0*f_p + f_pp) for
  // memory
  warm_factor =
      1.08333333333333333333 * (f - 2.0 * f_p + f_pp) * (f - 2.0 * f_p + f_pp);

  // Note : here i is not the current cell, but one plus (for upwinding)
  for (int i = 0; i < sampled_flux_at_nodes_size; ++i) {
    // clang-format on

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = f_m;
	f_m  = f;
	f    = f_p;
	f_p  = f_pp;
	const int curr_index = (i + 2) % sampled_flux_at_nodes_size;
	f_pp = 0.5 * (sampled_flux_at_nodes[curr_index] - alpha * uin[curr_index]);

#ifdef INTERNAL_DEBUG_
	printf("Index pp %d", curr_index);
    printf("Sampled mm at position %d is %f\n", i, f_mm);
    printf("Sampled m at position %d is %f\n", i, f_m);
    printf("Sampled p at position %d is %f\n", i, f_p);
    printf("Sampled pp at position %d is %f\n", i, f_pp);
#endif

	const Float fhat_1p = (0.3333333333333333334 * f  - 0.1666666666666666667 * f_mm  + 0.833333333333333333 * f_m  );
	const Float fhat_2p = (0.3333333333333333334 * f_m   - 0.1666666666666666667 * f_p  + 0.833333333333333333 * f  );
	const Float fhat_3p = (0.3333333333333333334 * f_pp  - 1.1666666666666666667 * f_p  + 1.833333333333333333 * f  );
    // clang-format on

#ifdef INTERNAL_DEBUG_
    printf("Sampled mm at position %d is %f\n", i, fhat_1n);
    printf("Sampled m at position %d is %f\n", i, fhat_2n);
    printf("Sampled p at position %d is %f\n", i, fhat_3n);
#endif

    // Smoothness indicators (\beta)
    const Float hot_factor = 1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                             (f_pp - 2.0 * f_p + f);
    const Float beta_1p = cold_factor + 0.25 * (f_mm - 4.0 * f_m + 3.0 * f) *
                                            (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2p = warm_factor + 0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3p = hot_factor + 0.25 * (f_pp - 4.0 * f_p + 3.0 * f) *
                                           (f_pp - 4.0 * f_p + 3.0 * f);
    cold_factor = warm_factor;
    warm_factor = hot_factor;

    // Construct non-linear, non-normalized weights
    const Float w_tilde_1p =
        WENO_GAMMA_3 / ((WENO_EPSILON + beta_1p) * (WENO_EPSILON + beta_1p));
    const Float w_tilde_2p =
        WENO_GAMMA_2 / ((WENO_EPSILON + beta_2p) * (WENO_EPSILON + beta_2p));
    const Float w_tilde_3p =
        WENO_GAMMA_1 / ((WENO_EPSILON + beta_3p) * (WENO_EPSILON + beta_3p));
    const Float w_sum_p = w_tilde_1p + w_tilde_2p + w_tilde_3p;

    // Construct WENO weights now
    const Float w1p = w_tilde_1p / w_sum_p;
    const Float w2p = w_tilde_2p / w_sum_p;
    const Float w3p = w_tilde_3p / w_sum_p;

    // Reconstruct numerical flux at cell boundary $u_{i-1/2}^{+}$
    // Put values in a new buffer to support reinitialization
    right_cell_boundary_flux[(i - 1 + sampled_flux_at_nodes_size) %
                             sampled_flux_at_nodes_size] =
        w1p * fhat_1p + w2p * fhat_2p + w3p * fhat_3p;
  }

  // clang-format off
  total_flux_at_center[0] = factor * (left_cell_boundary_flux[0] - left_cell_boundary_flux[sampled_flux_at_nodes_size - 1] + right_cell_boundary_flux[0] - right_cell_boundary_flux[sampled_flux_at_nodes_size - 1]);
  for (int i = 1; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
#ifdef INTERNAL_DEBUG_
    printf("Left Boundary flux at position %d + 0.5 is %f\n", i,
           left_cell_boundary_flux[i]);
    printf("Right Boundary flux at position %d - 0.5 is %f\n", i,
           right_cell_boundary_flux[i]);
#endif
    total_flux_at_center[i] =
        factor *
        (left_cell_boundary_flux[i] + right_cell_boundary_flux[i] -
         left_cell_boundary_flux[i - 1] - right_cell_boundary_flux[i - 1]);
  }
};

// Saves a modulo call, not much difference
template <class Float>
void weno5_FD_1D_all_novec_optimized_two(
    const Float uin[], const int uin_size, const Float sampled_flux_at_nodes[],
    const int sampled_flux_at_nodes_size, Float total_flux_at_center[],
    const int total_flux_at_center_size, Float boundary_flux[],
    const int boundary_flux_size, const Float alpha,
    const Float factor  // 1/ dx
) {
  // Manually unroll index at 0, 1, 2
  // clang-format off

  Float f_m =  0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 2] + alpha * uin[sampled_flux_at_nodes_size - 2]);
  Float f = 0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 1] + alpha * uin[sampled_flux_at_nodes_size - 1] ) ;
  Float f_p = 0.5 * (sampled_flux_at_nodes[0] + alpha * uin[0]);
  Float f_pp = 0.5 * (sampled_flux_at_nodes[1] + alpha * uin[1]);

  // Actually (f_mm - 2.0 * f_m + f), but here replaced by (f_m - 2.0*f + f_p) for memory
  Float cold_factor =
	1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p);
  // Actually (f_m - 2.0*f + f_p), but here replaced by (f - 2.0*f_p + f_pp) for memory
  Float warm_factor =
	1.08333333333333333333 * (f - 2.0 * f_p + f_pp) * (f - 2.0 * f_p + f_pp);

  for (int i = 0; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    // Skip flux splitting for now

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = f_m;
	f_m  = f;
	f    = f_p;
	f_p  = f_pp;
	const int curr_index = (i + 2) % sampled_flux_at_nodes_size;
	f_pp = 0.5 * (sampled_flux_at_nodes[curr_index]  + alpha * uin[curr_index]);

	const Float fhat_1n = (0.3333333333333333334 * f_mm  - 1.1666666666666666667 * f_m  + 1.833333333333333333 * f  );
	const Float fhat_2n = (0.3333333333333333334 * f_p   - 0.1666666666666666667 * f_m  + 0.833333333333333333 * f  );
	const Float fhat_3n = (0.3333333333333333334 * f     - 0.1666666666666666667 * f_pp + 0.833333333333333333 * f_p);
    // clang-format on

    // Smoothness indicators (\beta)
    const Float hot_factor = 1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                             (f_pp - 2.0 * f_p + f);
    const Float beta_1n = cold_factor + 0.25 * (f_mm - 4.0 * f_m + 3.0 * f) *
                                            (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2n = warm_factor + 0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3n = hot_factor + 0.25 * (f_pp - 4.0 * f_p + 3.0 * f) *
                                           (f_pp - 4.0 * f_p + 3.0 * f);
    cold_factor = warm_factor;
    warm_factor = hot_factor;

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
    boundary_flux[i] = w1n * fhat_1n + w2n * fhat_2n + w3n * fhat_3n;
  }

  // Indexing starts at 1, so modify preloads accordingly
  f_m = 0.5 * (sampled_flux_at_nodes[sampled_flux_at_nodes_size - 1] -
               alpha * uin[sampled_flux_at_nodes_size - 1]);
  f = 0.5 * (sampled_flux_at_nodes[0] - alpha * uin[0]);
  f_p = 0.5 * (sampled_flux_at_nodes[1] - alpha * uin[1]);
  f_pp = 0.5 * (sampled_flux_at_nodes[2] - alpha * uin[2]);

  // Actually (f_mm - 2.0 * f_m + f), but here replaced by (f_m - 2.0*f + f_p)
  // for memory
  cold_factor =
      1.08333333333333333333 * (f_m - 2.0 * f + f_p) * (f_m - 2.0 * f + f_p);
  // Actually (f_m - 2.0*f + f_p), but here replaced by (f - 2.0*f_p + f_pp) for
  // memory
  warm_factor =
      1.08333333333333333333 * (f - 2.0 * f_p + f_pp) * (f - 2.0 * f_p + f_pp);

  // Note : here i is the current cell
  // This version has one less modulo operation than the other case
  const int oneplus = sampled_flux_at_nodes_size + 1;
  for (int i = 1; i < oneplus; ++i) {
    // clang-format on

    // Reconstruct polynomials at (i + 1/2) location
    // $\hat{f}_{i+1/2}^{-} from f(u)$
    // clang-format off
	const Float f_mm = f_m;
	f_m  = f;
	f    = f_p;
	f_p  = f_pp;
	const int curr_index = (i + 2) % sampled_flux_at_nodes_size;
	f_pp = 0.5 * (sampled_flux_at_nodes[curr_index] - alpha * uin[curr_index]);

	const Float fhat_1p = (0.3333333333333333334 * f  - 0.1666666666666666667 * f_mm  + 0.833333333333333333 * f_m  );
	const Float fhat_2p = (0.3333333333333333334 * f_m   - 0.1666666666666666667 * f_p  + 0.833333333333333333 * f  );
	const Float fhat_3p = (0.3333333333333333334 * f_pp  - 1.1666666666666666667 * f_p  + 1.833333333333333333 * f  );
    // clang-format on

    // Smoothness indicators (\beta)
    const Float hot_factor = 1.08333333333333333333 * (f_pp - 2.0 * f_p + f) *
                             (f_pp - 2.0 * f_p + f);
    const Float beta_1p = cold_factor + 0.25 * (f_mm - 4.0 * f_m + 3.0 * f) *
                                            (f_mm - 4.0 * f_m + 3.0 * f);
    const Float beta_2p = warm_factor + 0.25 * (f_m - f_p) * (f_m - f_p);
    const Float beta_3p = hot_factor + 0.25 * (f_pp - 4.0 * f_p + 3.0 * f) *
                                           (f_pp - 4.0 * f_p + 3.0 * f);
    cold_factor = warm_factor;
    warm_factor = hot_factor;

    // Construct non-linear, non-normalized weights
    const Float w_tilde_1p =
        WENO_GAMMA_3 / ((WENO_EPSILON + beta_1p) * (WENO_EPSILON + beta_1p));
    const Float w_tilde_2p =
        WENO_GAMMA_2 / ((WENO_EPSILON + beta_2p) * (WENO_EPSILON + beta_2p));
    const Float w_tilde_3p =
        WENO_GAMMA_1 / ((WENO_EPSILON + beta_3p) * (WENO_EPSILON + beta_3p));
    const Float w_sum_p = w_tilde_1p + w_tilde_2p + w_tilde_3p;

    // Construct WENO weights now
    const Float w1p = w_tilde_1p / w_sum_p;
    const Float w2p = w_tilde_2p / w_sum_p;
    const Float w3p = w_tilde_3p / w_sum_p;

    // Reconstruct numerical flux at cell boundary $u_{i+1/2}^{-}$
    // flux_n
    boundary_flux[i - 1] += w1p * fhat_1p + w2p * fhat_2p + w3p * fhat_3p;
  }

  // clang-format off
  total_flux_at_center[0] = factor * (boundary_flux[0] - boundary_flux[sampled_flux_at_nodes_size - 1]);
  for (int i = 1; i < sampled_flux_at_nodes_size; ++i){
    // clang-format on
    total_flux_at_center[i] =
        factor * (boundary_flux[i] - boundary_flux[i - 1]);
  }
};
