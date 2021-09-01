#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

// #include "extrapolate_using_least_squares.hpp"
// #include "ExtrapolationConfig.hpp"

// int main(int argc, char *argv[]) {
//   using EC = lstsq::ExtrapolationConfiguration<3, 1>;

//   for (std::size_t i = 0UL; i < 9UL; ++i) {
//     std::cout << "( " << EC::y_mask[i] << ", " << EC::x_mask[i] << " )"
//               << " with sum " << EC::sum_mask[i] << std::endl;
//   }

//   return 0;
// }

#include "binomial_coefficient.hpp"

void test_binomial_coefficient() {
  static_assert(binomial_coefficient(8, 5) == 56, "Failed");
  static_assert(binomial_coefficient(7, 7) == 1, "Failed");
  static_assert(binomial_coefficient(3, 2) == 3, "Failed");
}

#include "least_squares.hpp"

void test_gauss_elimination() {
  // matrix in and out
  constexpr std::size_t n_coefficients(3UL), n_components(2UL),
      n_gridpoints(9UL);
  double LHS[n_coefficients][n_coefficients + n_components] = {
      {2.56, 0.86, 4.2, 1.964, 1.284},
      {0.86, 0.32, 1.4, 0.666, 0.45},
      {4.2, 1.4, 7., 3.22, 2.1},
  };
  double expected_sol[n_components][n_coefficients] = {{0.7, 0.2, 0.0},
                                                       {0.3, 0.6, 0.0}};
  double computed_sol[n_components][n_gridpoints];

  lstsq::gauss_elimination<n_coefficients, n_components, n_gridpoints>(
      LHS, computed_sol);

  double max_diff(0.0);
  for (std::size_t i_component = 0; i_component < n_components; ++i_component) {
    for (std::size_t i = 0; i < n_coefficients; i++) {
      max_diff = std::max(max_diff, std::abs(expected_sol[i_component][i] -
                                             computed_sol[i_component][i]));
    }
  }

  if (max_diff > 1e-14) {
    throw std::runtime_error("Lstsq solve not working");
  }

  std::cout << max_diff << std::endl;
}

int main(int argc, char* argv[]) { test_gauss_elimination(); }
