#pragma once

#include <cmath>
#include <type_traits>

namespace kernels {
  struct LinearKernel {
   public:
    static constexpr int kernel_start = -1;
    static constexpr int kernel_end = 1;
    static constexpr int kernel_size = kernel_end - kernel_start;

    // Kernel for |x| \leq 1
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 0), int>::type = 0>
    inline static Float _f(Float t) {
      return static_cast<Float>(1.0) - t;
    }

    template <typename Float>
    static Float eval(Float x) {
      Float t = std::fabs(x);
      int which_case = static_cast<int>(t) < 1 ? static_cast<int>(t) : 1;
      switch (which_case) {
        case 1:
          return 0;
        case 0:
          return _f<0>(t);
      }
      return 0;
    }
  };

}  // namespace kernels
