#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include "Utilities/NamedType.hpp"
#include "lstsq/ExtrapolationConfig.hpp"
#include "lstsq/least_squares.hpp"

namespace detail {

  template <typename ArrayIndexType, typename ShortArrayIndexType>
  struct LeastSquaresExtrapolationPointInfo {
    // indices of the point in x and y
    ArrayIndexType idx_x, idx_y;
    // start codes of the point in x and y
    ShortArrayIndexType start_x, start_y;
  };

  // Peel off the coefficients one by one
  // like in a Pascal's triangle
  // x^2 xy y^2
  //    x  y
  //     01
  // and put it into a matrix
  // Although inefficient, it also does the 0 case (basically puts a 1
  // in the last position)
  template <short ExtrapOrder, typename Float>
  void binomial_fill(Float LHS_row[], Float x, Float y) {
    int col_idx(0);
    // should be constexpr but oh well
    for (int power = ExtrapOrder; power > 0; --power) {
      // Go through the
      // multinomial terms of
      // x and y
      for (int x_power = power; x_power >= 0; --x_power, ++col_idx) {
        LHS_row[col_idx] =
            std::pow(x, x_power) * std::pow(y, (power - x_power));
      }
    }
    // Last row is 1.0 (don't bother doing any of the pow's)
    LHS_row[col_idx] = 1.0;
  }

  // TODO : This should be an overloaded operator within the class
  // TODO : This and the parallel fill should be static members of the
  // class for better SRP
  // Helper function for filling in the matrix
  // multiplies first * second
  template <short PointsInGrid, typename Float>
  void multiply(Float* __restrict__ in_one, Float* __restrict__ in_two,
                Float* __restrict__ out) {
    for (int i = 0; i < PointsInGrid; ++i) {
      out[i] = in_one[i] * in_two[i];
    }
  }

  // Assumes LHS is of the form [NumberOfCoefficients][PointsInGrid]
  // Incoming data of x, y is of the form [2][PointsInGrid]
  // Incoming data of mask is of the form [PointsInGrid] : separation of mask
  // and x,y is logical and not physical
  // Does [1, x, y, x*2, x*y, y*2, x*3, x^2*y, x*y^2, y^3 ...]
  // Concept : ExtrapOrder >= 1
  template <short ExtrapOrder, short NumberOfCoefficients, short PointsInGrid,
            typename Float>
  void binomial_fill_data_parallel(
      Float LHS[NumberOfCoefficients][PointsInGrid],
      Float position_data[2][PointsInGrid], Float mask[PointsInGrid]) {
    constexpr int X(0), Y(1);
    int src_idx(-1), tgt_idx(0);
    // Unpack first row of the triagnle
    // First row : zeroth power only
    std::copy_n(mask, PointsInGrid, LHS[0]);

    for (int i_order = 1; i_order <= ExtrapOrder; ++i_order) {
      // This fills in the i_order terms
      multiply<PointsInGrid>(LHS[src_idx + 1], position_data[X],
                             LHS[++tgt_idx]);
      for (int j_term = 0; j_term < i_order; ++j_term) {
        multiply<PointsInGrid>(LHS[++src_idx], position_data[Y],
                               LHS[++tgt_idx]);
      }
    }
  }

  // Assume A_T stored in row major format (so A is column major) for ease of
  // use
  // A (n_rows, n_columns)
  // A_T (n_columns, n_rows) : (n_coeffs, n_points)
  // B_jk = {A_T}_{ji}{A_T}{ki}
  // i : (0, n_rows), j, k : (0, n_cols)
  // which is the formulae used
  // A_T . A : (n_columns, n_columns)
  template <short NumberOfCoefficients, short PointsInGrid, typename Float>
  void multiply_by_transpose(
      Float const A_T[NumberOfCoefficients][PointsInGrid],
      Float out[NumberOfCoefficients][NumberOfCoefficients + 2]) {
    for (int j = 0; j < NumberOfCoefficients; ++j) {
      for (int k = 0; k < NumberOfCoefficients; ++k) {
        Float sum(0.0);
        for (int i = 0; i < PointsInGrid; ++i) {
          sum += A_T[j][i] * A_T[k][i];
        }
        out[j][k] = sum;
      }
    }
  }

  // Assume A_T stored in row major format (so A is column major) for ease of
  // use
  // A (n_rows, n_columns)
  // A_T (n_columns, n_rows) : (n_coeffs, n_points)
  // v : multidimensional, stored as [dim][n_points]
  // w_j = {A_T}_{ji}{v}{i}
  // i : (0, n_rows), j : (0, n_cols)
  // which is the formulae used
  template <short NumberOfCoefficients, short PointsInGrid, typename Float>
  void multiply_by_transpose_vec(
      Float const A_T[NumberOfCoefficients][PointsInGrid],
      Float const input_vec[2][PointsInGrid],
      Float out[NumberOfCoefficients][NumberOfCoefficients + 2]) {
    for (int j = 0; j < NumberOfCoefficients; ++j) {
      Float const* const A_T_j = A_T[j];
      for (int dim = 0; dim < 2; ++dim) {
        Float sum(0.0);
        Float const* const vec = input_vec[dim];
        for (int i = 0; i < PointsInGrid; ++i) {
          sum += A_T_j[i] * vec[i];
        }
        out[j][NumberOfCoefficients + dim] = sum;
      }
    }
  }

  template <typename ShortInt,  // Template parameter for boolean like
                                // indexing arrays : these can be bool or int,
                                // instantiated to be short int
            typename Float,     // Template parameter for floating point arrays
            class EC>           // Configuration of extrapolation, see
                                // detail::ExtrapolationConfiguration
  void extrapolate_using_least_squares_impl(
      ShortInt current_flag[], const int current_flag_size0,
      const int current_flag_size1, const ShortInt target_flag[],
      const int target_flag_size0, const int target_flag_size1, Float eta_x[],
      const int eta_x_size0, const int eta_x_size1, Float eta_y[],
      const int eta_y_size0, const int eta_y_size1, const Float grid_x[],
      const int grid_x_size0, const Float grid_y[], const int grid_y_size0) {
    // ExtrapConfig::

    // Vector of Gridinfos to calculate grid indices for points to be ex
    using PointInfo = LeastSquaresExtrapolationPointInfo<int, short>;
    std::vector<PointInfo> extrap_info;
    // Reserve conservatively
    constexpr float RESERVE_FACTOR(0.15);
    extrap_info.reserve(
        static_cast<std::size_t>(RESERVE_FACTOR * eta_x_size0 * eta_x_size1));

    // Reserve arrays of ShortInts and Floating types of ExtrapConfig size

    // Locally check always a 3 x 3 grid to first check
    // ShortInt local_flags[3 * 3];

    // For the least squares problem
    // sampled_points x n_coefficients
    // DONE : Should be storing LHS_T and rhs_T for better memory layout, but
    // filling gets affected
    Float LHS[EC::n_coefficients_to_extrapolate()][EC::n_points_in_grid()];
    // sampled_points x n_qts to be extapolated
    Float rhs[2][EC::n_points_in_grid()];
    Float flag_mask[EC::n_points_in_grid()];

    // Quantities after being hit by A^T
    // This includes both the vector and the matrix (should make no difference
    // at all, except for the memory)
    Float LHS_T_LHSrhs[EC::n_coefficients_to_extrapolate()]
                      [EC::n_coefficients_to_extrapolate() + 2];

    // Unfortunately, we need some additional memory to construct the local
    // binomial expansion coefficients
    // by careful refactorign, we should be able to reuse the above memory
    // created
    Float LHS_multinomial_local[EC::n_coefficients_to_extrapolate()][1];
    Float rhs_local[2][1];
    Float flag_mask_local[1] = {static_cast<Float>(1.0)};

    // Counter variable that tells how many more cells
    // need to be updated
    unsigned int counter(0);
    do {
      // Reset all quantities, inefficient to clear a vector everytime
      // think of reusing the structs
      extrap_info.clear();
      counter = 0;

      // # 1. select cells which has a neighboring temp_flag to be 0.
      for (int j = 0; j < current_flag_size0; ++j) {
        for (int i = 0; i < current_flag_size1; ++i) {
          // # Only if flag2 is one it needs to be considered, else break
          const unsigned int curr_index(j * current_flag_size1 + i);
          if (current_flag[curr_index] ^ target_flag[curr_index]) {
            ShortInt local_sum(0);
            constexpr int offset(-1);
#pragma unroll
            for (int jloc = offset; jloc < (offset + EC::search_width());
                 ++jloc) {
              const unsigned int local_row_index(curr_index +
                                                 jloc * current_flag_size1);
#pragma unroll
              for (int iloc = offset; iloc < (offset + EC::search_width());
                   ++iloc) {
                local_sum +=
                    current_flag[local_row_index + iloc] *
                    static_cast<ShortInt>(
                        EC::sum_mask[EC::search_width() * (jloc - offset) +
                                     (iloc - offset)]);
              }  // iloc
            }    // jloc

            if (local_sum) {
              ShortInt local_x_sum(0);
              ShortInt local_y_sum(0);
#pragma unroll
              for (int jloc = offset; jloc < (offset + EC::search_width());
                   ++jloc) {
                const unsigned int local_row_index(curr_index +
                                                   jloc * current_flag_size1);
#pragma unroll
                for (int iloc = offset; iloc < (offset + EC::search_width());
                     ++iloc) {
                  local_x_sum +=
                      current_flag[local_row_index + iloc] *
                      static_cast<ShortInt>(
                          EC::x_mask[EC::search_width() * (jloc - offset) +
                                     (iloc - offset)]);
                  local_y_sum +=
                      current_flag[local_row_index + iloc] *
                      static_cast<ShortInt>(
                          EC::y_mask[EC::search_width() * (jloc - offset) +
                                     (iloc - offset)]);
                }
              }

              // Remember this point info for the future
              extrap_info.push_back({i, j,
                                     EC::start_code(local_x_sum, local_sum),
                                     EC::start_code(local_y_sum, local_sum)});
              ++counter;
            }  // if local sum
          }    // if flag passes

        }  // i
      }    // j

      // std::printf("Counter is %u \n", counter);

      // # 2 Early return if no work to do
      if (!counter) {
        return;
      }

      /*
           for (auto& info : extrap_info) {
    std::printf("sum : ##, ystart : %d, xstart : %d at (%f, %f)\n",
                info.start_y, info.start_x, grid_x[info.idx_x],
                grid_y[info.idx_y]);
  }
       */

      // # TODO 2.1 Sort the array using code tricks for potentially
      // # faster access

      // # 3. For these cells solve the system and
      for (auto& info : extrap_info) {
        // 3.1 iterate in the local neighborhood and
        const int s_y(info.idx_y + static_cast<int>(info.start_y));
        const int e_y(s_y + EC::grid_size());

        const int s_x(info.idx_x + static_cast<int>(info.start_x));
        const int e_x(s_x + EC::grid_size());
        /* Data serial version without tranpose
            // Tells the index of the LHS array, or equivalently a counter
            // for the local grid
            std::size_t neighbor_idx(0UL);


                for (int jloc = s_y; jloc < e_y;
                                                  ++jloc) {
          const int local_row_index(jloc * current_flag_size1);
          Float y(grid_y[jloc]);
          for (int iloc = s_x; iloc < e_x; ++iloc, ++neighbor_idx) {
            const int grid_idx = local_row_index + iloc;
            Float x(grid_x[iloc]),
                mask(static_cast<Float>(current_flag[grid_idx]));

            // Fill the LHS matrix with various powers of x, y
            binomial_fill<EC::extrapolation_order()>(LHS[neighbor_idx], x, y);

            // Fill the rhs matrix with eta_1 and eta_2
            rhs[neighbor_idx][0] = eta_x[grid_idx];
            rhs[neighbor_idx][1] = eta_y[grid_idx];

            // Finally mask the LHS and RHS matries with correct points
            std::for_each(
                LHS[neighbor_idx],
                LHS[neighbor_idx] + EC::n_coefficients_to_extrapolate(),
                [mask](auto& d) { d *= mask; });
            rhs[neighbor_idx][0] *= mask;
            rhs[neighbor_idx][1] *= mask;
          }  // iloc
}    // jloc
        */
        // 3.2 load the data and mask it to construct LHS and RHS
        // matrices

        // Data parallel version : first gather into RHS then do a scatter
        // into LHS Requires an additional gridsize * gridsize amount of
        // memory to store mask float Tells the index of the LHS array, or
        // equivalently a counter for the local grid
        std::size_t neighbor_idx(0UL);

        for (int jloc = s_y; jloc < e_y; ++jloc) {
          const int local_row_index(jloc * current_flag_size1);
          const Float y(grid_y[jloc]);
          for (int iloc = s_x; iloc < e_x; ++iloc, ++neighbor_idx) {
            const int grid_idx = local_row_index + iloc;
            // Using the rhs as a temporary buffer to store x and y
            // with which we construct a
            rhs[0][neighbor_idx] = grid_x[iloc];
            rhs[1][neighbor_idx] = y;
            flag_mask[neighbor_idx] =
                static_cast<Float>(current_flag[grid_idx]);
          }
        }

        binomial_fill_data_parallel<EC::extrapolation_order(),
                                    EC::n_coefficients_to_extrapolate(),
                                    EC::n_points_in_grid()>(LHS, rhs,
                                                            flag_mask);

        /*
std::printf("(\n");
for (int row_idx = 0; row_idx < EC::n_coefficients_to_extrapolate();
     ++row_idx) {
  for (int col_idx = 0; col_idx < EC::n_points_in_grid(); ++col_idx) {
    std::printf("%f, ", LHS[row_idx][col_idx]);
  }
  std::printf("\n");
}
std::printf(")\n");
        */

        neighbor_idx = 0UL;
        for (int jloc = s_y; jloc < e_y; ++jloc) {
          const int local_row_index(jloc * current_flag_size1);
          for (int iloc = s_x; iloc < e_x; ++iloc, ++neighbor_idx) {
            const int grid_idx = local_row_index + iloc;
            // Using the rhs as a temporary buffer to store x and y
            // with which we construct a
            const Float mask(flag_mask[neighbor_idx]);
            rhs[0][neighbor_idx] = mask * eta_x[grid_idx];
            rhs[1][neighbor_idx] = mask * eta_y[grid_idx];
          }
        }

        /*
std::printf("(\n");
for (int row_idx = 0; row_idx < 2; ++row_idx) {
  for (int col_idx = 0; col_idx < EC::n_points_in_grid(); ++col_idx) {
    std::printf("%f, ", rhs[row_idx][col_idx]);
  }
  std::printf("\n");
}
std::printf(")\n");
*/

        // 3.3 Solve the least squares system
        multiply_by_transpose<EC::n_coefficients_to_extrapolate(),
                              EC::n_points_in_grid()>(LHS, LHS_T_LHSrhs);
        multiply_by_transpose_vec<EC::n_coefficients_to_extrapolate(),
                                  EC::n_points_in_grid()>(LHS, rhs,
                                                          LHS_T_LHSrhs);
        /*
std::printf("(\n");
for (int row_idx = 0; row_idx < EC::n_coefficients_to_extrapolate();
     ++row_idx) {
  for (int col_idx = 0;
       col_idx < EC::n_coefficients_to_extrapolate() + 2; ++col_idx) {
    std::printf("%f, ", LHS_T_LHSrhs[row_idx][col_idx]);
  }
  std::printf("\n");
}
std::printf(")\n");
         */

        // Gaussian elimination routine goes here
        lstsq::gauss_elimination<EC::n_coefficients_to_extrapolate(), 2,
                                 EC::n_points_in_grid()>(LHS_T_LHSrhs, rhs);

        /*
        std::printf("(\n");
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          for (int col_idx = 0; col_idx < EC::n_coefficients_to_extrapolate();
               ++col_idx) {
            std::printf("%f, ", rhs[row_idx][col_idx]);
          }
          std::printf("\n");
        }
        std::printf(")\n");
         */

        // Unpack the vector and multiply
        // 1. prepare the multinomial terms in the current location
        // Load the current x and y into the rhs vector
        // TODO : These all should be dimensions independent arrays :()
        rhs_local[0][0] = grid_x[info.idx_x];
        rhs_local[1][0] = grid_y[info.idx_y];
        binomial_fill_data_parallel<EC::extrapolation_order(),
                                    EC::n_coefficients_to_extrapolate(), 1>(
            LHS_multinomial_local, rhs_local, flag_mask_local);

        // 2. Do a data parallel multiplication with the results from LHS
        // stored in the first n_coefficients entry of RHS[2][n_points_in_grid]
        // TODO : Abstract away all these details
        Float eta_extrapolated(0.0);
        const int local_grid_idx(info.idx_y * eta_x_size1 + info.idx_x);
        for (int i = 0; i < EC::n_coefficients_to_extrapolate(); ++i) {
          eta_extrapolated += rhs[0][i] * LHS_multinomial_local[i][0];
        }
        eta_x[local_grid_idx] = eta_extrapolated;

        eta_extrapolated = 0.0;
        for (int i = 0; i < EC::n_coefficients_to_extrapolate(); ++i) {
          eta_extrapolated += rhs[1][i] * LHS_multinomial_local[i][0];
        }
        eta_y[local_grid_idx] = eta_extrapolated;

      }  // vector traversal

      // 4. Update the current flag because stuff there is already extrapolated
      for (auto& info : extrap_info) {
        const int local_grid_idx(info.idx_y * eta_x_size1 + info.idx_x);
        current_flag[local_grid_idx] = 1;
      }  // update flag
    } while (counter > 0);
  }

}  // namespace detail

template <class ShortInt, class Float>
void extrapolate_using_least_squares_till_first_order(
    ShortInt current_flag[], const int current_flag_size0,
    const int current_flag_size1, const ShortInt target_flag[],
    const int target_flag_size0, const int target_flag_size1, Float eta_x[],
    const int eta_x_size0, const int eta_x_size1, Float eta_y[],
    const int eta_y_size0, const int eta_y_size1, const Float grid_x[],
    const int grid_x_size0, const Float grid_y[], const int grid_y_size0) {
  detail::extrapolate_using_least_squares_impl<
      ShortInt, Float,
      lstsq::extrapolation_configuration_t<lstsq::GridSize<3>,
                                           lstsq::ExtrapolationOrder<1>,
                                           lstsq::Dimensions<2>>>(
      current_flag, current_flag_size0, current_flag_size1, target_flag,
      target_flag_size0, target_flag_size1, eta_x, eta_x_size0, eta_x_size1,
      eta_y, eta_y_size0, eta_y_size1, grid_x, grid_x_size0, grid_y,
      grid_y_size0);
}

template <class ShortInt, class Float>
void extrapolate_using_least_squares_till_second_order(
    ShortInt current_flag[], const int current_flag_size0,
    const int current_flag_size1, const ShortInt target_flag[],
    const int target_flag_size0, const int target_flag_size1, Float eta_x[],
    const int eta_x_size0, const int eta_x_size1, Float eta_y[],
    const int eta_y_size0, const int eta_y_size1, const Float grid_x[],
    const int grid_x_size0, const Float grid_y[], const int grid_y_size0) {
  detail::extrapolate_using_least_squares_impl<
      ShortInt, Float,
      lstsq::extrapolation_configuration_t<lstsq::GridSize<3>,
                                           lstsq::ExtrapolationOrder<2>,
                                           lstsq::Dimensions<2>>>(
      current_flag, current_flag_size0, current_flag_size1, target_flag,
      target_flag_size0, target_flag_size1, eta_x, eta_x_size0, eta_x_size1,
      eta_y, eta_y_size0, eta_y_size1, grid_x, grid_x_size0, grid_y,
      grid_y_size0);
}
