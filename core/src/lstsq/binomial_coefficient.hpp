#pragma once

#include <type_traits>

namespace detail {

  template <typename T>
  constexpr T binomial_coefficient_recursive_impl(const T n,
                                                  const T k) noexcept {
    // edge cases first
    return
        // deals with 0 choose 0 or (k | C | k) case
        (k == T(0) || n == k) ? T(1) :
                              // return 0 if expansino coefficient if 0
            (n == T(0) ? T(0) :
                       // handle everything else
                 (binomial_coefficient_recursive_impl(n - 1, k - 1) +
                  binomial_coefficient_recursive_impl(n - 1, k)));
  }

}  // namespace detail

template <typename T1,                               // First type
          typename T2,                               // Second Type
          typename TC = std::common_type_t<T1, T2>>  // Common type
constexpr TC binomial_coefficient(const T1 n, const T2 k) noexcept {
  return detail::binomial_coefficient_recursive_impl(static_cast<TC>(n),
                                                     static_cast<TC>(k));
}
