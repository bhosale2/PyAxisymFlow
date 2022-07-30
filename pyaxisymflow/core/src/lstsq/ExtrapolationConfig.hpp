#pragma once
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "../Utilities/NamedTemplate.hpp"
#include "binomial_coefficient.hpp"

namespace lstsq {

  namespace detail {

    using IndexType = unsigned short;

    // Compute the binomial coefficient during compile-time
    inline float round_up_to_next_even_number(float f) {
      return std::ceil(0.5 * f) * 2.0;
    }

    // http://prosepoetrycode.potterpcs.net/2015/07/a-simple-constexpr-power-function-c/
    template <typename T>
    constexpr T ipow(T num, unsigned int pow) {
      return (pow >= sizeof(unsigned int) * 8)
                 ? 0
                 : pow == 0 ? 1 : num * ipow(num, pow - 1);
    }

    // Helpers to generate mask and sum arrays
    // TODO : Extend to 3D
    template <IndexType GridSize, IndexType Dimensions = 2>
    struct SumMaskGenerator {
      static constexpr IndexType MaskSize = ipow(GridSize, Dimensions);
      static constexpr IndexType generate(IndexType x) {
        return (x == (MaskSize - 1) / 2) ? static_cast<IndexType>(0)
                                         : static_cast<IndexType>(1);
      }
    };

    // Helpers to generate mask and sum arrays
    template <IndexType GridSize, IndexType Dimensions = 2>
    struct XMaskGenerator {
      static constexpr IndexType MaskSize = ipow(GridSize, Dimensions);
      static constexpr IndexType generate(IndexType x) { return x % GridSize; }
    };

    template <IndexType GridSize, IndexType Dimensions = 2>
    struct YMaskGenerator {
      static constexpr IndexType MaskSize = ipow(GridSize, Dimensions);
      static constexpr IndexType generate(IndexType x) { return x / GridSize; }
    };

    template <IndexType GridSize, IndexType Dimensions = 2>
    struct ZMaskGenerator {
      static constexpr IndexType MaskSize = ipow(GridSize, Dimensions);
      static constexpr IndexType generate(IndexType x) {
        return x / ipow(GridSize, 2);
      }
    };

    // TODO : Concept check : is it a Generator Type?
    template <class Generator>
    struct SequenceGenerator {
      template <IndexType... Ts>
      static constexpr std::array<IndexType, sizeof...(Ts)> generate_impl(
          std::integer_sequence<IndexType, Ts...>) {
        return {{Generator::generate(Ts)...}};
      }
      static constexpr auto generate() {
        return generate_impl(
            std::make_integer_sequence<IndexType, Generator::MaskSize>{});
      }
    };

    template <IndexType LeastSquaresGridSize,  // Gives the local square grid
                                               // size used to extrapolate
                                               // values
              IndexType ExtrapolationOrder,    // Gives order of accuracy
                                               // information
              IndexType Dimensions = 2>        // Dimensionality of the problem
    struct ExtrapolationConfiguration {
      // int because finally we end up using an int
      static constexpr int grid_size() noexcept {
        return static_cast<int>(LeastSquaresGridSize);
      }

      static constexpr int extrapolation_order() noexcept {
        return static_cast<int>(ExtrapolationOrder);
      }

      static constexpr std::size_t n_points_in_grid() noexcept {
        return static_cast<std::size_t>(LeastSquaresGridSize) *
               static_cast<std::size_t>(LeastSquaresGridSize);
      }
      // Gives number of coefficients
      // For example in case of linear extrapolation (ExtrapolationOrder == 1)
      // this fits a*x + b*y + c = eta, hence number of coefficients is 3
      // In general, the formulae is
      // (Dimensions + ExtrapolationOrder | C | ExtrapolationOrder)---( . |C | .
      // ) is the combinatory choose operation

      // Proof :
      // https://en.wikipedia.org/wiki/Multinomial_theorem#Number_of_multinomial_coefficients
      // For (x_1 + x_2 + x_3 .... + x_m)^n
      // the number of coefficients NC(.) is (n + m - 1 | C | m - 1)
      // We need the following sum:
      // \sum_{j=0}^{Ord} NC((x_1 + x_2 + x_3 .... + x_{dim}) ^ j)
      // \sum_{j=0}^{Ord} (j + dim - 1 | C | dim - 1)
      // Using the formula for sum over upper index found here:
      // https://proofwiki.org/wiki/Sum_of_Binomial_Coefficients_over_Upper_Index
      // The formula eventually reads
      // (order + dim | C | dim)
      static constexpr std::size_t n_coefficients_to_extrapolate() noexcept {
        return binomial_coefficient(
            static_cast<std::size_t>(ExtrapolationOrder) +
                static_cast<std::size_t>(Dimensions),
            static_cast<std::size_t>(Dimensions));
      }

      // Tells how much to scale the code based on the grid size
      template <typename Int>
      static inline constexpr Int code_prefactor() {
        return static_cast<Int>((LeastSquaresGridSize + 1)) /
               static_cast<Int>(2);
      }

      // Code is the weighted average of the sums and positional sums
      // The if statement only exists to support degenerate configurations
      // occuring when there are many (>5) adjacent grid points with information
      // within them
      template <typename Int>
      static inline Int code(Int positional_sum, Int total_sum) noexcept {
        float temp = (positional_sum > static_cast<Int>(4))
                         ? round_up_to_next_even_number(
                               static_cast<float>(positional_sum))
                         : static_cast<float>(positional_sum);
        return static_cast<Int>(2.0 * temp / total_sum);
      }

      // To make the start code Gridsize independent, we need to do some extra
      // operations
      // As it involves a subtract, typically don't pass in an unsigned int
      template <typename Int>
      static inline Int start_code(Int code) noexcept {
        return static_cast<Int>(code_prefactor<Int>() * code -
                                static_cast<Int>(2 * LeastSquaresGridSize)) /
               static_cast<Int>(2);
      }

      // For the most useful case of GridSize == 3, this boils down to
      // just code - 3 (the code prefactor in that case is 2).
      // Hence declare a specialization (although compiler should optimize the
      // previous calls away)

      template <typename Int>
      static inline Int start_code(Int positional_sum, Int total_sum) noexcept {
        return start_code(code(positional_sum, total_sum));
      }

      static_assert(
          n_points_in_grid() > n_coefficients_to_extrapolate() + 1UL,
          "Not enough grid points to fit coefficients to the order you are "
          "looking "
          "for : either increase the grid size or reduce the order "
          "requirements");
      // static_assert((LeastSquaresGridSize + 1) % 4 == 0,
      //               "Grid size needs to be a of size 3, 7, 11... for now");

      // Note : we only consider a local 3 x 3 cell to determine where to place
      // the grid for extrapolation. Hence this value is hardcoded below:
      static constexpr auto search_width() { return static_cast<IndexType>(3); }

      using MaskType =
          decltype(SequenceGenerator<
                   XMaskGenerator<search_width(), Dimensions>>::generate());
      // static MaskStorageType sum_mask;
      // static MaskStorageType x_mask;
      // static MaskStorageType y_mask;

      // Static constexpr members
      constexpr static MaskType sum_mask = SequenceGenerator<
          SumMaskGenerator<search_width(), Dimensions>>::generate();
      constexpr static MaskType x_mask = SequenceGenerator<
          XMaskGenerator<search_width(), Dimensions>>::generate();
      constexpr static MaskType y_mask = SequenceGenerator<
          YMaskGenerator<search_width(), Dimensions>>::generate();
    };

    /*
    template <IndexType LeastSquaresGridSize, // Gives the local square grid
                                            // size used to extrapolate values
            IndexType ExtrapolationOrder,   // Gives order of accuracy
            // information
            IndexType Dimensions> // Dimensionality of the problem
    typename ExtrapolationConfiguration<LeastSquaresGridSize,
    ExtrapolationOrder, Dimensions>::MaskStorageType
      ExtrapolationConfiguration<LeastSquaresGridSize, ExtrapolationOrder,
                                 Dimensions>::sum_mask =
          SequenceGenerator<
              SumMaskGenerator<LeastSquaresGridSize, Dimensions>>::generate();
    */

    template <IndexType LeastSquaresGridSize,  // Gives the local square grid
                                               // size used to extrapolate
                                               // values
              IndexType ExtrapolationOrder,    // Gives order of accuracy
                                               // information
              IndexType Dimensions>            // Dimensionality of the problem
    constexpr typename ExtrapolationConfiguration<
        LeastSquaresGridSize, ExtrapolationOrder, Dimensions>::MaskType
        ExtrapolationConfiguration<LeastSquaresGridSize, ExtrapolationOrder,
                                   Dimensions>::sum_mask;

    template <IndexType LeastSquaresGridSize,  // Gives the local square grid
                                               // size used to extrapolate
                                               // values
              IndexType ExtrapolationOrder,    // Gives order of accuracy
                                               // information
              IndexType Dimensions>            // Dimensionality of the problem
    constexpr typename ExtrapolationConfiguration<
        LeastSquaresGridSize, ExtrapolationOrder, Dimensions>::MaskType
        ExtrapolationConfiguration<LeastSquaresGridSize, ExtrapolationOrder,
                                   Dimensions>::x_mask;

    template <IndexType LeastSquaresGridSize,  // Gives the local square grid
                                               // size used to extrapolate
                                               // values
              IndexType ExtrapolationOrder,    // Gives order of accuracy
                                               // information
              IndexType Dimensions>            // Dimensionality of the problem
    constexpr typename ExtrapolationConfiguration<
        LeastSquaresGridSize, ExtrapolationOrder, Dimensions>::MaskType
        ExtrapolationConfiguration<LeastSquaresGridSize, ExtrapolationOrder,
                                   Dimensions>::y_mask;
  }  // namespace detail

  template <detail::IndexType Int>
  using GridSize =
      utilities::NamedTemplate<std::integral_constant<detail::IndexType, Int>,
                               struct GridSizeTag>;

  template <detail::IndexType Int>
  using ExtrapolationOrder =
      utilities::NamedTemplate<std::integral_constant<detail::IndexType, Int>,
                               struct ExtrapOrderTag>;

  template <detail::IndexType Int>
  using Dimensions =
      utilities::NamedTemplate<std::integral_constant<detail::IndexType, Int>,
                               struct DimensionsTag>;

  // Alias templates can't be specailzed, thus adding one more indirection which
  // uses a struct to mimic
  namespace detail {
    template <typename, typename, typename>
    struct make_extrapolation_configuration_impl;

    template <IndexType grid_size, IndexType extrap_order, IndexType dim>
    struct make_extrapolation_configuration_impl<
        GridSize<grid_size>, ExtrapolationOrder<extrap_order>,
        Dimensions<dim>> {
      using type = ExtrapolationConfiguration<grid_size, extrap_order, dim>;
    };

    template <IndexType grid_size, IndexType extrap_order, IndexType dim>
    struct make_extrapolation_configuration_impl<
        ExtrapolationOrder<extrap_order>, GridSize<grid_size>,
        Dimensions<dim>> {
      using type = ExtrapolationConfiguration<grid_size, extrap_order, dim>;
    };

  }  // namespace detail

  /*
  template <detail::IndexType grid_size, detail::IndexType extrap_order,
            detail::IndexType dim>
  using make_extrapolation_configuration =
      typename detail::make_extrapolation_configuration_impl<
        GridSize<grid_size>, extrap_order, dim>::type;

  template
  <GridSize<grid_size>, ExtrapolationOrder<extrap_order>, Dimensions<dim>>
    using make_extrapolation_configuration =
        typename detail::make_extrapolation_configuration_impl<
            GridSize<grid_size>, extrap_order, dim>::type;
*/

  template <typename A, typename B, typename C>
  using extrapolation_configuration_t =
      typename detail::make_extrapolation_configuration_impl<A, B, C>::type;
};  // namespace lstsq
