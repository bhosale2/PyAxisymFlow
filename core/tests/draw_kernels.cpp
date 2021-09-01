#include <algorithm>
#include <cassert>
#include <experimental/tuple>
#include <iostream>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#include "particle_kernels.hpp"

namespace plt = matplotlibcpp;

template <typename KernelType>
void draw_kernel(bool verbose = false) {
  std::size_t n_points(200UL);
  std::vector<double> x(n_points, 0.0);
  double half_length = 3.0;
  double dx = 2.0 * half_length / n_points;
  std::iota(x.begin(), x.end(), 0.0);
  std::for_each(x.begin(), x.end(), [dx, half_length](double& d) {
    d *= dx;
    d -= half_length;
  });
  /*
  std::iota(x.begin(), x.end(), -(double)(n_points)*0.5);
  std::for_each(x.begin(), x.end(), [spacing](double &d) { d *= spacing; });
  */

  std::vector<double> kernel(n_points, 0.0);
  std::transform(x.begin(), x.end(), kernel.begin(),
                 [](double d) { return KernelType::eval(d); });

  if (verbose) {
    for (std::size_t i = 0UL; i < n_points; ++i) {
      std::cout << x[i] << " " << kernel[i] << std::endl;
    }
  }

  plt::figure(1);
  plt::plot(x, kernel, "-bo");
}

template <typename KernelType>
void test_kernel_wrapper_1D(bool verbose = false) {
  using WrappedKernelType = kernels::KernelWrapper<KernelType>;

  double scaled_distance_to_least_adjacent_node(0.3);
  double scaled_distance[KernelType::kernel_size];
  for (int i = KernelType::kernel_start; i < KernelType::kernel_end; ++i)
    scaled_distance[i - KernelType::kernel_start] =
        std::fabs(scaled_distance_to_least_adjacent_node + (double)i);

  double weights[KernelType::kernel_size];
  WrappedKernelType::interpolate(scaled_distance, weights);

  if (verbose) {
    std::cout << "Weights" << std::endl;
    std::for_each(weights, weights + KernelType::kernel_size,
                  [](double d) { std::cout << d << ", "; });
    std::cout << std::endl << "Eval" << std::endl;
    std::for_each(scaled_distance, scaled_distance + KernelType::kernel_size,
                  [](double d) { std::cout << KernelType::eval(d) << ", "; });
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < KernelType::kernel_size; ++i) {
    assert(std::fabs(KernelType::eval(scaled_distance[i]) - weights[i]) < 1e-6);
  }
}

template <typename KernelType>
void test_kernel_wrapper_2D(bool verbose = false) {
  using WrappedKernelType = kernels::KernelWrapper<KernelType>;

  double scaled_distance_to_least_adjacent_node_x(0.3);
  double scaled_distance_to_least_adjacent_node_y(0.4);
  double scaled_distance[KernelType::kernel_size][2];
  for (int i = KernelType::kernel_start; i < KernelType::kernel_end; ++i) {
    scaled_distance[i - KernelType::kernel_start][0] =
        std::fabs(scaled_distance_to_least_adjacent_node_x + (double)i);
    scaled_distance[i - KernelType::kernel_start][1] =
        std::fabs(scaled_distance_to_least_adjacent_node_y + (double)i);
  }

  double weights[KernelType::kernel_size][2];
  WrappedKernelType::interpolate(scaled_distance, weights);

  if (verbose) {
    std::cout << "Weights" << std::endl;
    std::for_each(weights, weights + KernelType::kernel_size, [](double d[2]) {
      std::cout << d[0] << ", " << d[1] << ", ";
    });
    std::cout << std::endl << "Eval" << std::endl;
    std::for_each(scaled_distance, scaled_distance + KernelType::kernel_size,
                  [](double d[2]) {
                    std::cout << KernelType::eval(d[0]) << ", "
                              << KernelType::eval(d[1]) << ", ";
                  });
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < KernelType::kernel_size; ++i) {
    assert(std::fabs(KernelType::eval(scaled_distance[i][0]) - weights[i][0]) <
           1e-6);
    assert(std::fabs(KernelType::eval(scaled_distance[i][1]) - weights[i][1]) <
           1e-6);
  }
}

template <typename KernelType>
void test_kernel() {
  draw_kernel<KernelType>();
  test_kernel_wrapper_1D<KernelType>();
  test_kernel_wrapper_2D<KernelType>();
  plt::show();
}

int main(int argc, char* argv[]) {
  using KernelTypes =
      std::tuple<kernels::LinearKernel, kernels::QuadraticSpline, kernels::MP4,
                 kernels::MP6, kernels::YangSmoothThreePointKernel>;
  using YangKernelTypes = std::tuple<kernels::YangSmoothThreePointKernel>;
  std::experimental::apply(
      [](auto... curr_kernel) {
        using expander = int[];
        (void)expander{0, ((void)test_kernel<decltype(curr_kernel)>(), 0)...};
      },
      YangKernelTypes{});
  return 0;
}
