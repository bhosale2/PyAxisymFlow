#pragma once

#include <cmath>
#include <type_traits>

namespace kernels {

  /* Kernel definition : copy paste directly in desmos
        https://www.desmos.com/calculator/rmqpmdjdmo
        0\left\{2<\left|x\right|\right\}
        \frac{1}{2}(2-\left|x\right|)^{2}(1-\left|x\right|)\left\{1<\left|x\right|\le2\right\}
        1-\frac{5}{2}x^{2}+\frac{3}{2}\left|x\right|^{3}\left\{0<\left|x\right|\le1\right\}
  */
  struct MP4 {
   public:
    static constexpr int kernel_start = -2;
    static constexpr int kernel_end = 2;
    static constexpr int kernel_size = kernel_end - kernel_start;

    // Kernel for |x| \leq 1
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 0), int>::type = 0>
    inline static Float _f(Float t) {
      return static_cast<Float>(1.0) +
             t * t * (static_cast<Float>(-2.5) + static_cast<Float>(1.5) * t);
    }

    // Kernel for 1 < |x| \leq 2
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 1), int>::type = 0>
    inline static Float _f(Float t) {
      return static_cast<Float>(2.0) +
             t * (static_cast<Float>(-4.0) +
                  t * (static_cast<Float>(2.5) - static_cast<Float>(0.5) * t));
    }

    template <typename Float>
    static Float eval(Float x) {
      Float t = std::fabs(x);
      int which_case = static_cast<int>(t) < 2 ? static_cast<int>(t) : 2;
      switch (which_case) {
        case 2:
          return 0;
        case 1:
          return _f<1>(t);
        case 0:
          return _f<0>(t);
      }
      return 0;
    }
  };

}  // namespace kernels
