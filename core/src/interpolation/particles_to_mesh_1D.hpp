#pragma once
#include <cmath>
#include <cstdio>

namespace detail {

  /*
   1D version of p2m
   */
  template <class Float, class InterpolationKernel>
  void particles_to_mesh_impl(const Float particle_positions[],
                              const int particle_positions_size,
                              const Float input_field_at_particle_positions[],
                              const int input_field_at_particle_positions_size,
                              Float output_field[], const int output_field_size,
                              const Float delta_x) {
    Float weights[InterpolationKernel::kernel_size];
    Float scaled_distance[InterpolationKernel::kernel_size];
    // Should be memset
    for (int j_mesh = 0; j_mesh < output_field_size; ++j_mesh) {
      output_field[j_mesh] = 0.0;
    }

    for (int i_particle = 0; i_particle < particle_positions_size;
         ++i_particle) {
      const Float x_particle = particle_positions[i_particle] / delta_x;
      const int highest_mesh_index_to_particle =
          static_cast<int>(std::ceil(x_particle));
      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        scaled_distance[i - InterpolationKernel::kernel_start] =
            std::fabs(x_particle -
                      static_cast<Float>(highest_mesh_index_to_particle + i));
      }

      InterpolationKernel::interpolate(scaled_distance, weights);

      const Float input = input_field_at_particle_positions[i_particle];
      bool last_particle =
          // (i_particle == (particle_positions_size - 1)) ? true : false;
          (i_particle == (0)) ? true : false;
      if (last_particle) {
        printf("Particle at %f has redistribution index at : ", x_particle);
      }
      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        const int mesh_index =
            (highest_mesh_index_to_particle + i + output_field_size) %
            output_field_size;
        if (last_particle) {
          printf("%d, %f\t", mesh_index,
                 weights[i - InterpolationKernel::kernel_start]);
        }
        output_field[mesh_index] +=
            weights[i - InterpolationKernel::kernel_start] * input;
      }
      if (last_particle)
        printf("\n");
    }
    /*
    // Assumption is that particles are close to initial mesh position
    // so we only search for a short span around
    constexpr int SEARCH_WIDTH = 5;
    for (int i_mesh = 0; i_mesh < output_field_size; ++i_mesh) {
      Float accumulated(0.0);
      for (int j_p = -SEARCH_WIDTH; j_p < SEARCH_WIDTH; ++j_p) {
        const int particle_index =
            (i_mesh - SEARCH_WIDTH + output_field_size) % output_field_size;
        const Float x_particle = particle_positions[particle_index] / delta_x;
      }
    }
    */
  };

  template <class Float, class InterpolationKernel>
  void particles_to_mesh_with_offset_impl(
      const Float particle_positions[], const int particle_positions_size,
      const Float input_field_at_particle_positions[],
      const int input_field_at_particle_positions_size, Float output_field[],
      const int output_field_size, const Float delta_x) {
    Float weights[InterpolationKernel::kernel_size];
    Float scaled_distance[InterpolationKernel::kernel_size];

    // Should be memset
    for (int j_mesh = 0; j_mesh < output_field_size; ++j_mesh) {
      output_field[j_mesh] = 0.0;
    }

    for (int i_particle = 0; i_particle < particle_positions_size;
         ++i_particle) {
      const Float x_particle = particle_positions[i_particle] / delta_x;
      // Calculated virtual mesh (aka if not offset by 0.5)
      const Float least_virtual_mesh_index_to_particle = std::floor(x_particle);
      // Takes a call as to whether the particles is to the left or right of the
      // actual mesh sitting in between the virtual mesh
      const int highest_mesh_index_to_particle =
          static_cast<int>(x_particle >=
                           (least_virtual_mesh_index_to_particle + 0.5)) +
          static_cast<int>(least_virtual_mesh_index_to_particle);

      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        // We once again add 0.5 here because the mesh indices are offset by
        // 0.5*h the highest mesh index simply provides the index
        // (non-dimensional distance) to closest mesh point
        scaled_distance[i - InterpolationKernel::kernel_start] =
            std::fabs(x_particle - (highest_mesh_index_to_particle + 0.5 + i));
      }

      InterpolationKernel::interpolate(scaled_distance, weights);

      const Float input = input_field_at_particle_positions[i_particle];
#ifdef INTERNAL_DEBUG_
      bool last_particle =
          (i_particle == (particle_positions_size - 1)) ? true : false;
      if (last_particle) {
        printf("Particle at %f has redistribution index at : ", x_particle);
      }
#endif

      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        const int mesh_index =
            (highest_mesh_index_to_particle + i + output_field_size) %
            output_field_size;
#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("%d, %f\t", mesh_index,
                 weights[i - InterpolationKernel::kernel_start]);
        }
#endif
        output_field[mesh_index] +=
            weights[i - InterpolationKernel::kernel_start] * input;
      }
#ifdef INTERNAL_DEBUG_
      if (last_particle)
        printf("\n");
#endif
    }
  }
}  // namespace detail
