#include <algorithm>
#include <iostream>
#include <numeric>

#include "extrapolate_using_least_squares.hpp"
#include "least_squares.hpp"

int main(int argc, char* argv[]) {
  constexpr int n_rows(9);
  constexpr int n_columns(4);
  constexpr int input_size(n_rows * n_columns);
  // constexpr int output_size(n_rows * n_rows);
  constexpr int n_output = n_columns;
  constexpr int output_size(n_output * n_output);
  float input[input_size], output[output_size];
  std::fill(input, input + input_size, 0.0f);
  std::fill(output, output + output_size, 0.0f);

  std::iota(input, input + input_size, 0.0f);
  std::for_each(input, input + input_size, [](float& f) { f /= 100.0f; });

  // dot_with_pretranspose(input, n_rows, n_columns, output);
  // dot_with_pretranspose_simd(input, n_rows, n_columns, output);
  dot_with_pretranspose_simd_nohadd(input, n_rows, output);

  std::cout << "Input" << std::endl;
  std::cout << "[" << std::endl;
  for (int j = 0; j < n_rows; ++j) {
    for (int i = 0; i < n_columns; ++i) {
      const int curr_index(j * n_columns + i);
      std::cout << input[curr_index] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;

  // Display A^T * A
  std::cout << "Output" << std::endl;
  std::cout << "[" << std::endl;
  for (int j = 0; j < n_output; ++j) {
    for (int i = 0; i < n_output; ++i) {
      const int curr_index(j * n_rows + i);
      std::cout << output[curr_index] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}
/*
int main(int argc, char *argv[]) {
  using ExtrapolationType =
      detail::ExtrapolationConfiguration<3,  // size of grid
                                         2>; // order of accuracy
  static_assert(ExtrapolationType::n_points_in_grid() == 9, "fail");
  std::cout << "n_coefficients : "
            << ExtrapolationType::n_coefficients_to_extrapolate() << std::endl;
  std::cout << "code_prefactor : " << ExtrapolationType::code_prefactor<int>()
            << std::endl;
  int code = ExtrapolationType::code(4, 3);
  std::cout << "code : " << code << std::endl;
  int start_code = ExtrapolationType::start_code(code);
  std::cout << "start_code : " << start_code << std::endl;
  start_code = ExtrapolationType::start_code(4, 3);
  std::cout << "overloaded start_code : " << start_code << std::endl;
  return 0;
}
*/
