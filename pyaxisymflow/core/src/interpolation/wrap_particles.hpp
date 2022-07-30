#pragma once
#include <algorithm>

namespace detail {
  enum WrappingStrategy { FirstN, All };

  template <WrappingStrategy>
  struct WrappingLimitChooser;

  // Choose this option to
  // Wrap particles around in a halo region of 10 points from start and end
  // or the size, whichever is minimal
  template <>
  struct WrappingLimitChooser<WrappingStrategy::FirstN> {
    static constexpr int n_particles_to_wrap = 10;
    inline static int left_limit(const int size) noexcept {
      return std::min(n_particles_to_wrap, size);
    }

    inline static int right_limit(const int left_limit,
                                  const int size) noexcept {
      return std::max(left_limit, size - n_particles_to_wrap);
    }
  };

  // Choose this option to wrap all particles around the domain irrespective of
  // the indices : useful if the domain's small (testing purposes)
  template <>
  struct WrappingLimitChooser<WrappingStrategy::All> {
    static constexpr int n_particles_to_wrap = 10;
    inline static int left_limit(const int size) noexcept { return size; }
    inline static int right_limit(const int /*left_limit*/,
                                  const int /*size*/) noexcept {
      return 0;
    }
  };

  template <WrappingStrategy WrapStrategy, class Float>
  void wrap_particles_around_1D_domain_impl(Float particle_positions[],
                                            const int particle_positions_size,
                                            const Float domain_start,
                                            const Float domain_end) {
    // const int left_end = std::min(10, particle_positions_size);
    // const int right_start = std::max(left_end, particle_positions_size - 10);

    // const int left_end = particle_positions_size;
    // const int right_start = 0;

    using LimitChooser = WrappingLimitChooser<WrapStrategy>;
    const int left_end = LimitChooser::left_limit(particle_positions_size);
    const int right_start =
        LimitChooser::right_limit(left_end, particle_positions_size);

    const Float domain_length = domain_end - domain_start;

    for (int i = 0; i < left_end; ++i) {
      const Float pp = particle_positions[i];
      particle_positions[i] = (pp < domain_start) ? pp + domain_length : pp;
    }
    for (int i = right_start; i < particle_positions_size; ++i) {
      const Float pp = particle_positions[i];
      particle_positions[i] = (pp > domain_end) ? pp - domain_length : pp;
    }
  }
}  // namespace detail

/*
  2D versions follow
 */

namespace detail {

  template <class Float>
  void wrap_particles_around_2D_domain_in_x_impl(
      Float particle_positions_x[], const int particle_positions_x_size0,
      const int particle_positions_x_size1, const Float domain_start_x,
      const Float domain_end_x) {
    // Since we are doing only x, its possible to do it row by row
    // delegation is then done to the 1D version
    for (int j = 0; j < particle_positions_x_size0; ++j) {
      const int current_row_idx = j * particle_positions_x_size1;
      wrap_particles_around_1D_domain_impl<WrappingStrategy::FirstN>(
          &particle_positions_x[current_row_idx], particle_positions_x_size1,
          domain_start_x, domain_end_x);
    }
  }

  template <class Float>
  void wrap_particles_around_2D_domain_in_y_impl(
      Float particle_positions_y[], const int particle_positions_y_size0,
      const int particle_positions_y_size1, const Float domain_start_y,
      const Float domain_end_y) {
    // Only the top and bottom rows need to be checked for particles going out
    // of domain.

    // 1. We get these limits using the limit chooser
    using TopAndBottomLimitChooser =
        WrappingLimitChooser<WrappingStrategy::FirstN>;
    const int bottom_end =
        TopAndBottomLimitChooser::left_limit(particle_positions_y_size0);
    const int top_start = TopAndBottomLimitChooser::right_limit(
        bottom_end, particle_positions_y_size0);

    // 2. We ship it off to 1D with an all particles strategy
    // 2.1 Do the bottom particles
    for (int j = 0; j < bottom_end; ++j) {
      const int current_row_idx = j * particle_positions_y_size1;
      wrap_particles_around_1D_domain_impl<WrappingStrategy::All>(
          &particle_positions_y[current_row_idx], particle_positions_y_size1,
          domain_start_y, domain_end_y);
    }

    // 2.1 Do the top particles
    for (int j = top_start; j < particle_positions_y_size0; ++j) {
      const int current_row_idx = j * particle_positions_y_size1;
      wrap_particles_around_1D_domain_impl<WrappingStrategy::All>(
          &particle_positions_y[current_row_idx], particle_positions_y_size1,
          domain_start_y, domain_end_y);
    }
  }
}  // namespace detail
