#pragma once
#include <type_traits>
#include <utility>

namespace kernels {

  // Wraps a kernels internal functions f and returns a templated
  // function "interpolate"
  // TODO : Type trait checking for kernel
  // TODO : Can be CRTPed as well : does it offer any benefits apart
  // from code-reuse (first 3 lines only)
  template <typename InterpolationKernel>
  struct KernelWrapper {
    static constexpr int kernel_start = InterpolationKernel::kernel_start;
    static constexpr int kernel_end = InterpolationKernel::kernel_end;
    static constexpr int kernel_size = InterpolationKernel::kernel_size;

    static_assert(kernel_size % 2 == 0, "Kernel size is not even");
    static constexpr int half_kernel_size = kernel_size / 2;
    // static constexpr int half_kernel_size = 2;

    /* Say we pass in an array of kernel width like so, for kernels of total
       width even. To calculate the weights from the distance function, we
       then need to operate the following functions

         x are grid points
         o is the particle position

         stencil:	 x-----x-----x--o--x-----x-----x
         iterate:	 0     1     2     3     4     5
         functio:   _f2   _f1   _f0   _f0   _f1   _f2

                 To achieve this, we can wrap up the kernel like so
        */
    template <
        int Iteration,   // Iteration number from [0, half_kernel_size)
        typename Float,  // Floating point type
        typename std::enable_if<(Iteration < half_kernel_size), int>::type = 0>
    static inline Float _interp(Float scaled_distance) {
      return InterpolationKernel::template _f<(half_kernel_size - Iteration -
                                               1)>(scaled_distance);
    }

    template <
        int Iteration,  // Iteration number from [half_kernel_size, kernel_size)
        typename Float,  // Floating point type
        typename std::enable_if<(Iteration >= half_kernel_size), int>::type = 0>
    static inline Float _interp(Float scaled_distance) {
      return InterpolationKernel::template _f<(Iteration - half_kernel_size)>(
          scaled_distance);
    }

    // 1D version
    template <typename Float, int... Is>
    static void interpolate_impl(const Float in_scaled_distance[kernel_size],
                                 Float out_interp_weights[kernel_size],
                                 std::integer_sequence<int, Is...>) {
      using expander = int[];
      (void)expander{0, ((void)(out_interp_weights[Is] =
                                    _interp<Is>(in_scaled_distance[Is])),
                         0)...};
    }

    template <typename Float>
    static inline void interpolate(const Float in_scaled_distance[kernel_size],
                                   Float out_interp_weights[kernel_size]) {
      interpolate_impl(in_scaled_distance, out_interp_weights,
                       std::make_integer_sequence<int, kernel_size>{});
    }

    // 2D version
    template <typename Float, int... Is>
    static void interpolate_impl(const Float in_scaled_distance[kernel_size][2],
                                 Float out_interp_weights[kernel_size][2],
                                 std::integer_sequence<int, Is...>) noexcept {
      using expander = int[];
      (void)expander{0, ((void)(out_interp_weights[Is][0] =
                                    _interp<Is>(in_scaled_distance[Is][0]),
                                out_interp_weights[Is][1] =
                                    _interp<Is>(in_scaled_distance[Is][1])),
                         0)...};
    }

    template <typename Float>
    static inline void interpolate(
        const Float in_scaled_distance[kernel_size][2],
        Float out_interp_weights[kernel_size][2]) noexcept {
      interpolate_impl(in_scaled_distance, out_interp_weights,
                       std::make_integer_sequence<int, kernel_size>{});
    }
  };
}  // namespace kernels
