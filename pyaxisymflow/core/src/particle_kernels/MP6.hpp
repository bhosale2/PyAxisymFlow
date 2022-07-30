#pragma once

#include <cmath>
#include <type_traits>

namespace kernels {

  // clang-format off
  /* Kernel definition : copy paste directly in desmos
	 https://www.desmos.com/calculator/6vdtspxua3
	 0\left\{\left|x\right|>3\right\}
	 \left(-\frac{1}{24}(\left|x\right|-2)(\left|x\right|-3)^{3}(5\left|x\right|-8)\right)\left\{2<\left|x\right|\le3\right\}
	 \frac{1}{24}(\left|x\right|-1)(\left|x\right|-2)\left(25\left|x\right|^{3}-114x^{2}+153\left|x\right|-48\right)\left\{1<\left|x\right|\le2\right\}
	 -\frac{1}{12}(\left|x\right|-1)\left(25x^{4}-38\left|x\right|^{3}-3x^{2}+12\left|x\right|+12\right)\left\{0<\left|x\right|\le1\right\}
  */
  // clang-format on
  struct MP6 {
   public:
    static constexpr int kernel_start = -3;
    static constexpr int kernel_end = 3;
    static constexpr int kernel_size = kernel_end - kernel_start;

    // Kernel for |x| \leq 1
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 0), int>::type = 0>
    inline static Float _f(Float t) {
      // Cannot factor
      return 0.08333333333333333 * (1.0 - t) *
             (12.0 + t * (12.0 + t * (-3.0 + t * (-38.0 + 25.0 * t))));
    }

    // Kernel for 1 < |x| \leq 2
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 1), int>::type = 0>
    inline static Float _f(Float t) {
      // Cannot factor
      return 0.041666666666666666 * (t - 1.0) * (t - 2.0) *
             (-48.0 + t * (153.0 + t * (-114.0 + 25.0 * t)));
    }

    // Kernel for 2 < |x| \leq 3
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 2), int>::type = 0>
    inline static Float _f(Float t) {
      // https://en.cppreference.com/w/cpp/numeric/math/pow
      // pow always type promotoed to double
      // so unused here
      Float temp(t - 3.0);
      return 0.041666666666666666 * (2.0 - t) * (5.0 * t - 8.0) * temp * temp *
             temp;
    }

    template <typename Float>
    static Float eval(Float x) {
      Float t = std::fabs(x);
      int which_case = static_cast<int>(t) < 3 ? static_cast<int>(t) : 3;
      switch (which_case) {
        case 3:
          return 0;
        case 2:
          return _f<2>(t);
        case 1:
          return _f<1>(t);
        case 0:
          return _f<0>(t);
      }
      return 0;
    }
  };
}  // namespace kernels
