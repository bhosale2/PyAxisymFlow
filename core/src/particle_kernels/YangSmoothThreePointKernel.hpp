#pragma once
#include <cmath>
#include <type_traits>

namespace kernels {

  /* Kernel definition : copy paste directly in desmos
         https://www.desmos.com/calculator/an4a0jicw0
  \frac{17}{48}+\frac{\sqrt{3}\pi}{108}+\frac{\left|x\right|}{4}-\frac{\left|x\right|^{2}}{4}+\frac{1-2\left|x\right|}{16}\sqrt{-12\left|x\right|^{2}+12\left|x\right|+1}-\frac{\sqrt{3}}{12}\arcsin\left(\frac{\sqrt{3}}{2}(2\left|x\right|-1)\right)\left\{\left|x\right|\le1\right\}
  \frac{55}{48}-\frac{\sqrt{3}\pi}{108}-\frac{13\left|x\right|}{12}+\frac{\left|x\right|^{2}}{4}+\frac{2\left|x\right|-3}{48}\sqrt{-12\left|x\right|^{2}+36\left|x\right|-23}+\frac{\sqrt{3}}{36}\arcsin\left(\frac{\sqrt{3}}{2}(2\left|x\right|-3)\right)\left\{1<\left|x\right|\le2\right\}
  0\left\{\left|x\right|>2\right\}


  This follows from eq(20) in the paper
  A smoothing technique for discrete delta functions with application to
  immersed boundary method in moving boundary simulations Xiaolei Yang a, Xing
  Zhang a, Zhilin Li b, Guo-Wei He a,*
   */
  struct YangSmoothThreePointKernel {
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
      // >>> 17./48. + np.sqrt(3.0)*np.pi/108.0
      // 0.4045499823398394

      // >>> np.sqrt(3.0)/2.0
      // 0.8660254037844386
      // >>> np.sqrt(3.0)/12.0
      // 0.14433756729740643

      return 0.4045499823398394 + 0.25 * t * (1.0 - t) +
             (0.0625 - 0.125 * t) * std::sqrt(1.0 + 12.0 * t * (1.0 - t)) -
             0.14433756729740643 *
                 std::asin(0.8660254037844386 * (2.0 * t - 1.0));
    }

    // Kernel for 1 < |x| \leq 2
    template <
        unsigned short NonDimensionalWidth,  // Width of sampling
        typename Float,                      // Floating point type
        typename std::enable_if<(NonDimensionalWidth == 1), int>::type = 0>
    inline static Float _f(Float t) {
      // >>> 55./48. - np.sqrt(3.0) * np.pi / 108.0
      // 1.0954500176601605

      // >>> np.sqrt(3.0)/36.0
      // 0.04811252243246881

      return 1.0954500176601605 + t * (0.25 * t - 1.0833333333333333) +
             (0.04166666666666666 * t - 0.0625) *
                 std::sqrt(12.0 * t * (3.0 - t) - 23.0) +
             0.04811252243246881 *
                 std::asin(0.8660254037844386 * (2.0 * t - 3.0));
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
