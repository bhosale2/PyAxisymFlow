#pragma once
#include <cmath>

namespace lstsq {

  // Cannot use any unsigned ints because we are doing a negative indexing
  // routine here. By templating short, we narrow down the size of matrices that
  // can be put in, as shuold be the case
  // TODO : Concept check : NGridPoints > NCoefficients
  template <short NCoefficients, short NComponents, short NGridPoints,
            typename Float>
  void gauss_elimination(Float A[NCoefficients][NCoefficients + NComponents],
                         Float sol[NComponents][NGridPoints]) {
    short n_columns(NCoefficients + NComponents);

    for (short i = 0; i < NCoefficients; i++) {
      // Search for maximum in this column
      // Although pivoting is not needed in our use case
      // its best to do it to prevent precision loss
      short max_row = i;
      {
        Float max_element = abs(A[i][i]);
        for (short k = i + 1; k < NCoefficients; k++) {
          if (abs(A[k][i]) > max_element) {
            max_element = std::abs(A[k][i]);
            max_row = k;
          }
        }
      }

      // Swap maximum row with current row (column by column)
      {
        for (short k = i; k < n_columns; k++) {
          Float tmp = A[max_row][k];
          A[max_row][k] = A[i][k];
          A[i][k] = tmp;
        }
      }

      // Make all rows below this one 0 in current column
      for (short k = i + 1; k < NCoefficients; k++) {
        Float c = -A[k][i] / A[i][i];
        A[k][i] = static_cast<Float>(0.0);
        for (short j = (i + 1); j < n_columns; j++) {
          A[k][j] += c * A[i][j];
        }
      }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    for (short i_component = 0; i_component < NComponents; ++i_component) {
      for (short i = (NCoefficients - 1); i >= 0; --i) {
        sol[i_component][i] = A[i][NCoefficients + i_component] / A[i][i];
        for (short k = i - 1; k >= 0; --k) {
          A[k][NCoefficients + i_component] -= A[k][i] * sol[i_component][i];
        }
      }
    }
  }
}  // namespace lstsq

// #include <tmmintrin.h>
/*
void solve_least_squares(float *A, float* b){
  // Assume


}
*/
/*
// Assume A_T stored in rowwise format for ease of use
void dot_with_pretranspose(float *A_T, int n_rows, int n_cols, float *out) {
  for (int j = 0; j < n_rows; ++j) {
    const int first_offset = j * n_cols;
    for (int k = 0; k < n_rows; ++k) {
      const int second_offset = k * n_cols;
      float sum(0.0f);
      for (int i = 0; i < n_cols; ++i) {
        // std::cout << "( " << i << ", " << j << ", " << k << ")" << std::endl;
        sum += (A_T[first_offset + i] * A_T[second_offset + i]);
      }
      const int output_offset = j * n_rows + k;
      out[output_offset] = sum;
    }
  }
}

float hsum_ps_sse3(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// Still naive
// Assumes a shape of (12, 3) for A or (3 , 12) for A.T
void dot_with_pretranspose_simd(float *A_T, int n_rows, int n_cols,
                                float *out) {
  for (int j = 0; j < n_rows; ++j) {
    const int first_offset = j * n_cols;
    for (int k = 0; k < n_rows; ++k) {
      const int second_offset = k * n_cols;
      __m128 sum(_mm_set_ps1(0.0));
      for (int i = 0; i < n_cols; i += 4) {
        const __m128 pre_part = _mm_load_ps(&A_T[first_offset + i]);
        const __m128 post_part = _mm_load_ps(&A_T[second_offset + i]);
        sum = _mm_add_ps(sum, _mm_mul_ps(pre_part, post_part));
      }
      const int output_offset = j * n_rows + k;
      out[output_offset] = hsum_ps_sse3(sum);
    }
  }
}

// Assumes a shape of (9, 4) for A or alternatively (4, 9) for A.T
// A is stored in row-major format
// The last column is assumed to be unused
void dot_with_pretranspose_simd_nohadd(float *A, int n_rows, float *out) {
  __m128 first_output_row(_mm_set_ps1(0.0));
  __m128 second_output_row(_mm_set_ps1(0.0));
  __m128 third_output_row(_mm_set_ps1(0.0));
  __m128 fourth_output_row(_mm_set_ps1(0.0));
  for (int i = 0; i < n_rows; ++i) {
    __m128 row_A = _mm_load_ps(&A[4 * i]);
    __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);
    __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);
    __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);
    __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);

    first_output_row = _mm_add_ps(first_output_row, _mm_mul_ps(brod1, row_A));
    second_output_row = _mm_add_ps(second_output_row, _mm_mul_ps(brod2, row_A));
    third_output_row = _mm_add_ps(third_output_row, _mm_mul_ps(brod3, row_A));
    fourth_output_row = _mm_add_ps(fourth_output_row, _mm_mul_ps(brod4, row_A));
  }
  _mm_store_ps(&out[0], first_output_row);
  _mm_store_ps(&out[4], second_output_row);
  _mm_store_ps(&out[8], third_output_row);
  _mm_store_ps(&out[12], fourth_output_row);
}

void dot_with_pretranspose_simd_nohadd_v2(float *A, int n_rows, float *out) {
  __m128 first_output_row(_mm_set_ps1(0.0));
  __m128 second_output_row(_mm_set_ps1(0.0));
  __m128 third_output_row(_mm_set_ps1(0.0));
  __m128 fourth_output_row(_mm_set_ps1(0.0));
  for (int i = 0; i < n_rows; i += 4) {
    __m128 row_1 = _mm_load_ps(&A[4 * i]);
    __m128 row_2 = _mm_load_ps(&A[4 * (i + 1)]);
    __m128 row_3 = _mm_load_ps(&A[4 * (i + 2)]);
    __m128 row_4 = _mm_load_ps(&A[4 * (i + 3)]);

    for (int j = 0; j < 4; ++j) {

      __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);
      __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);
      __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);
      __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);
    }
    first_output_row = _mm_add_ps(first_output_row, _mm_mul_ps(brod1, row_A));
    second_output_row = _mm_add_ps(second_output_row, _mm_mul_ps(brod2, row_A));
    third_output_row = _mm_add_ps(third_output_row, _mm_mul_ps(brod3, row_A));
    fourth_output_row = _mm_add_ps(fourth_output_row, _mm_mul_ps(brod4, row_A));
  }
  _mm_store_ps(&out[0], first_output_row);
  _mm_store_ps(&out[4], second_output_row);
  _mm_store_ps(&out[8], third_output_row);
  _mm_store_ps(&out[12], fourth_output_row);
}
void __internal__matvec(float *A, float *b) {
  // Assumption is that A is (n, 12)
  // b is (n, 1)
  // for most common cases n = 3 or n = 6
  for (int j = 0; j < 12; j += 4) {
    const __m128 first_b = __mm_load_ps(b[j]);
    for (int i = 0; i < 4; i += 2) {
      const __m128 first = __mm_load_ps(A[0]);
      const __m128 second = __mm_load_ps(A[0]);
    }
  }
}
*/
