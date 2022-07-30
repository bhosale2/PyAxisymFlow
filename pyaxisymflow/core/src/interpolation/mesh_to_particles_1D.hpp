#pragma once
#include <cmath>
#include <cstdio>

/*
 1D version of m2p
 */
namespace detail {

  template <class Float, class InterpolationKernel>
  void mesh_to_particles_impl(const Float input_field[],
                              const int input_field_size,
                              const Float particle_positions[],
                              const int particle_positions_size,
                              Float output_field[], const int output_field_size,
                              const Float delta_x) {
    Float weights[InterpolationKernel::kernel_size];
    Float scaled_distance[InterpolationKernel::kernel_size];

    // assert(particle_positions_size == output_field_size);

    // Do not block just yet
    // Now we only work with non-dimensional distance below
    for (int i_particle = 0; i_particle < particle_positions_size;
         ++i_particle) {
      Float accumulated(0.0);
      const Float x_particle = particle_positions[i_particle] / delta_x;
      // const Float least_virtual_mesh_index_to_particle =
      // std::floor(x_particle); const Float highest_mesh_index_to_particle =
      // Diego's code here
      // Assumption is that particles wrap around periodic domain so somewhere
      // there should be an fmod
      const int highest_mesh_index_to_particle =
          static_cast<int>(std::ceil(x_particle));

      // Compute interpolation mesh distances
      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        scaled_distance[i - InterpolationKernel::kernel_start] =
            std::fabs(x_particle -
                      static_cast<Float>(highest_mesh_index_to_particle + i));
      }

      // Assign weights
      // DONE Kernel independent way (eval without losing performance)
      InterpolationKernel::interpolate(scaled_distance, weights);

      // for (int i = InterpolationKernel::kernel_start;
      //      i < InterpolationKernel::kernel_end; ++i) {
      //   weights[i - InterpolationKernel::kernel_start] =
      //       InterpolationKernel::eval(
      //           scaled_distance[i - InterpolationKernel::kernel_start]);
      // }

      // Fix for non-periodic, use one-sided interpolation
      // const bool potentially_out_of_domain =
      //     ((highest_mesh_index_to_particle +
      //     InterpolationKernel::kernel_start)
      //     <
      //      0) ||
      //     (highest_mesh_index_to_particle + InterpolationKernel::kernel_end >
      //      input_field_size);

      bool last_particle =
          (i_particle == (particle_positions_size - 1)) ? true : false;
      // (i_particle == (0)) ? true : false;

      if (last_particle) {
        printf("Particle at %f has redistribution index at : ", x_particle);
      }

      // Compute weighted sum, adjusting for periodicity
      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        const int mesh_index =
            (highest_mesh_index_to_particle + i + input_field_size) %
            input_field_size;
        accumulated += weights[i - InterpolationKernel::kernel_start] *
                       input_field[mesh_index];
        if (last_particle) {
          printf("%d, %f\t", mesh_index,
                 weights[i - InterpolationKernel::kernel_start]);
        }
      }
      output_field[i_particle] = accumulated;

      if (last_particle)
        printf("\n");
    }
  };

  // 1D version
  template <class Float, class InterpolationKernel>
  void mesh_to_particles_with_offset_impl(
      const Float input_field[], const int input_field_size,
      const Float particle_positions[], const int particle_positions_size,
      Float output_field[], const int output_field_size, const Float delta_x) {
    Float weights[InterpolationKernel::kernel_size];
    Float scaled_distance[InterpolationKernel::kernel_size];
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

#ifdef INTERNAL_DEBUG_
      bool last_particle =
          (i_particle == (particle_positions_size - 1)) ? true : false;
      if (last_particle) {
        printf("Particle at %f has redistribution index at : ", x_particle);
      }
#endif

      Float accumulated(0.0);
      for (int i = InterpolationKernel::kernel_start;
           i < InterpolationKernel::kernel_end; ++i) {
        const int mesh_index =
            (highest_mesh_index_to_particle + i + input_field_size) %
            input_field_size;
        accumulated += weights[i - InterpolationKernel::kernel_start] *
                       input_field[mesh_index];
#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("%d, %f\t", mesh_index,
                 weights[i - InterpolationKernel::kernel_start]);
        }
#endif
      }
      output_field[i_particle] = accumulated;
#ifdef INTERNAL_DEBUG_
      if (last_particle)
        printf("\n");
#endif
    }
  }
}  // namespace detail
