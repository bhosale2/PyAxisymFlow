#pragma once
#include <cmath>
#include <cstdio>

namespace detail {

  /*********************************************************************
   *
   *       2D version of m2p implementation
   *       for periodic cases
   *
   *********************************************************************/
  template <class Float, class InterpolationKernel>
  void particles_to_mesh_with_offset_impl_2D(
      const Float particle_positions_x[], const int particle_positions_x_size0,
      const int particle_positions_x_size1, const Float particle_positions_y[],
      const int particle_positions_y_size0,
      const int particle_positions_y_size1,
      const Float input_field_at_particle_positions[],
      const int input_field_at_particle_positions_size0,
      const int input_field_at_particle_positions_size1,
      Float output_field_at_mesh[], const int output_field_at_mesh_size0,
      const int output_field_at_mesh_size1, const Float delta_x,
      const Float delta_y) {
    Float weights[InterpolationKernel::kernel_size][2];
    Float scaled_distance[InterpolationKernel::kernel_size][2];

    // 0. Clear the output field before filling in
    {
      // for (int j_mesh = 0; j_mesh < output_field_at_mesh_size0; ++j_mesh)
      //   for (int i_mesh = 0; i_mesh < output_field_at_mesh_size1; ++i_mesh)
      //     output_field_at_mesh[i_mesh + j_mesh * output_field_at_mesh_size0]
      //     =
      //         static_cast<Float>(0.0);

      std::fill(output_field_at_mesh,
                output_field_at_mesh +
                    output_field_at_mesh_size0 * output_field_at_mesh_size1,
                static_cast<Float>(0.));
    }

    // 1. Loop over particles and compute weights to the mesh
    // Delegate any checks in the python kernel
    const int stride = particle_positions_x_size1;
#ifdef INTERNAL_DEBUG_
    printf("size0 : %d\n", particle_positions_x_size0);
    printf("size1 : %d\n", particle_positions_x_size1);
#endif

    for (int j_particle = 0; j_particle < particle_positions_x_size0;
         ++j_particle) {
      for (int i_particle = 0; i_particle < particle_positions_x_size1;
           ++i_particle) {
#ifdef INTERNAL_DEBUG_
        bool last_particle =
            (i_particle == (particle_positions_x_size1 - 1)) ? true : false;
#endif

        const int curr_particle_id = i_particle + j_particle * stride;

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("( %d, %d, %d)", i_particle, j_particle, curr_particle_id);
        }
#endif

        const Float curr_particle_position[2] = {
            particle_positions_x[curr_particle_id] / delta_x,
            particle_positions_y[curr_particle_id] / delta_y};

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf(
              "Particle at x=%f, y=%f has following redistribution index :\n",
              curr_particle_position[0], curr_particle_position[1]);
        }
#endif

        const Float least_virtual_mesh_index_to_particle[2] = {
            std::floor(curr_particle_position[0]),
            std::floor(curr_particle_position[1])};

        // Takes a call as to whether the particles is to the left or right of
        // the actual mesh sitting in between the virtual mesh
        const int highest_mesh_index_to_particle[2] = {
            static_cast<int>(curr_particle_position[0] >=
                             (least_virtual_mesh_index_to_particle[0] + 0.5)) +
                static_cast<int>(least_virtual_mesh_index_to_particle[0]),
            static_cast<int>(curr_particle_position[1] >=
                             (least_virtual_mesh_index_to_particle[1] + 0.5)) +
                static_cast<int>(least_virtual_mesh_index_to_particle[1])};

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("Highest virtual mesh idx=%d, idy=%d :\n",
                 highest_mesh_index_to_particle[0],
                 highest_mesh_index_to_particle[1]);
        }
#endif

        for (int i = InterpolationKernel::kernel_start;
             i < InterpolationKernel::kernel_end; ++i) {
          // We once again add 0.5 here because the mesh indices are offset by
          // 0.5*h the highest mesh index simply provides the index
          // (non-dimensional distance) to closest mesh point
          scaled_distance[i - InterpolationKernel::kernel_start][0] =
              std::fabs(curr_particle_position[0] -
                        (highest_mesh_index_to_particle[0] + 0.5 + i));
          scaled_distance[i - InterpolationKernel::kernel_start][1] =
              std::fabs(curr_particle_position[1] -
                        (highest_mesh_index_to_particle[1] + 0.5 + i));
        }

        InterpolationKernel::interpolate(scaled_distance, weights);

        const Float input = input_field_at_particle_positions[curr_particle_id];

        // for (int sx = InterpolationKernel::kernel_start;
        //      sx < InterpolationKernel::kernel_end; ++sx) {
        //   partial_sums[sx - InterpolationKernel::kernel_start] =
        //       weights[sx - InterpolationKernel::kernel_start][0] * input;
        // }

        for (int sy = InterpolationKernel::kernel_start;
             sy < InterpolationKernel::kernel_end; ++sy) {
          const int scaled_y_mesh_index =
              output_field_at_mesh_size1 * ((highest_mesh_index_to_particle[1] +
                                             sy + output_field_at_mesh_size0) %
                                            output_field_at_mesh_size0);
          const Float y_weight =
              weights[sy - InterpolationKernel::kernel_start][1];
          for (int sx = InterpolationKernel::kernel_start;
               sx < InterpolationKernel::kernel_end; ++sx) {
            const int mesh_index =
                // This is the x_mesh index
                ((highest_mesh_index_to_particle[0] + sx +
                  output_field_at_mesh_size1) %
                 output_field_at_mesh_size1) +
                //
                scaled_y_mesh_index;
            const Float weight =
                y_weight * weights[sx - InterpolationKernel::kernel_start][0];
            output_field_at_mesh[mesh_index] += weight * input;
          }
        }
      }
    }
  }

  /*********************************************************************
   *
   *       2D version of m2p implementation
   *       for unbounded cases
   *
   *********************************************************************/
  template <class Float, class InterpolationKernel>
  void particles_to_mesh_with_offset_impl_2D_unbounded(
      const Float particle_positions_x[], const int particle_positions_x_size0,
      const int particle_positions_x_size1, const Float particle_positions_y[],
      const int particle_positions_y_size0,
      const int particle_positions_y_size1,
      const Float input_field_at_particle_positions[],
      const int input_field_at_particle_positions_size0,
      const int input_field_at_particle_positions_size1,
      Float output_field_at_mesh[], const int output_field_at_mesh_size0,
      const int output_field_at_mesh_size1, const Float delta_x,
      const Float delta_y) {
    Float weights[InterpolationKernel::kernel_size][2];
    Float scaled_distance[InterpolationKernel::kernel_size][2];

    // 0. Clear the output field before filling in
    {
      // for (int j_mesh = 0; j_mesh < output_field_at_mesh_size0; ++j_mesh)
      //   for (int i_mesh = 0; i_mesh < output_field_at_mesh_size1; ++i_mesh)
      //     output_field_at_mesh[i_mesh + j_mesh * output_field_at_mesh_size0]
      //     =
      //         static_cast<Float>(0.0);

      std::fill(output_field_at_mesh,
                output_field_at_mesh +
                    output_field_at_mesh_size0 * output_field_at_mesh_size1,
                static_cast<Float>(0.));
    }

    // 1. Loop over particles and compute weights to the mesh
    // Delegate any checks in the python kernel
    const int stride = particle_positions_x_size1;
#ifdef INTERNAL_DEBUG_
    printf("size0 : %d\n", particle_positions_x_size0);
    printf("size1 : %d\n", particle_positions_x_size1);
#endif

    for (int j_particle = 0; j_particle < particle_positions_x_size0;
         ++j_particle) {
      for (int i_particle = 0; i_particle < particle_positions_x_size1;
           ++i_particle) {
#ifdef INTERNAL_DEBUG_
        bool last_particle =
            (i_particle == (particle_positions_x_size1 - 1)) ? true : false;
#endif

        const int curr_particle_id = i_particle + j_particle * stride;

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("( %d, %d, %d)", i_particle, j_particle, curr_particle_id);
        }
#endif

        const Float curr_particle_position[2] = {
            particle_positions_x[curr_particle_id] / delta_x,
            particle_positions_y[curr_particle_id] / delta_y};

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf(
              "Particle at x=%f, y=%f has following redistribution index :\n",
              curr_particle_position[0], curr_particle_position[1]);
        }
#endif

        const Float least_virtual_mesh_index_to_particle[2] = {
            std::floor(curr_particle_position[0]),
            std::floor(curr_particle_position[1])};

        // Takes a call as to whether the particles is to the left or right of
        // the actual mesh sitting in between the virtual mesh
        const int highest_mesh_index_to_particle[2] = {
            static_cast<int>(curr_particle_position[0] >=
                             (least_virtual_mesh_index_to_particle[0] + 0.5)) +
                static_cast<int>(least_virtual_mesh_index_to_particle[0]),
            static_cast<int>(curr_particle_position[1] >=
                             (least_virtual_mesh_index_to_particle[1] + 0.5)) +
                static_cast<int>(least_virtual_mesh_index_to_particle[1])};

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("Highest virtual mesh idx=%d, idy=%d :\n",
                 highest_mesh_index_to_particle[0],
                 highest_mesh_index_to_particle[1]);
        }
#endif

        for (int i = InterpolationKernel::kernel_start;
             i < InterpolationKernel::kernel_end; ++i) {
          // We once again add 0.5 here because the mesh indices are offset by
          // 0.5*h the highest mesh index simply provides the index
          // (non-dimensional distance) to closest mesh point
          scaled_distance[i - InterpolationKernel::kernel_start][0] =
              std::fabs(curr_particle_position[0] -
                        (highest_mesh_index_to_particle[0] + 0.5 + i));
          scaled_distance[i - InterpolationKernel::kernel_start][1] =
              std::fabs(curr_particle_position[1] -
                        (highest_mesh_index_to_particle[1] + 0.5 + i));
        }

        InterpolationKernel::interpolate(scaled_distance, weights);

        const Float input = input_field_at_particle_positions[curr_particle_id];

        // for (int sx = InterpolationKernel::kernel_start;
        //      sx < InterpolationKernel::kernel_end; ++sx) {
        //   partial_sums[sx - InterpolationKernel::kernel_start] =
        //       weights[sx - InterpolationKernel::kernel_start][0] * input;
        // }

        // Check if the particle is out of the domain, or so close to the edges
        // that interpolation may not be possible

        // If on the edge, then do only one-sided interpolation with whatever
        // values we have,  with the weights unchanged
        // - This doesn't conserve the moment perfectly
        const bool is_out_or_almost_out_of_domain =
            ((highest_mesh_index_to_particle[0] +
              InterpolationKernel::kernel_start) < 0) ||
            ((highest_mesh_index_to_particle[1] +
              InterpolationKernel::kernel_start) < 0) ||
            ((highest_mesh_index_to_particle[0] +
              InterpolationKernel::kernel_end) > output_field_at_mesh_size1) ||
            ((highest_mesh_index_to_particle[1] +
              InterpolationKernel::kernel_end) > output_field_at_mesh_size0);
        {
          const int start[2] = {
              is_out_or_almost_out_of_domain
                  ? std::max(InterpolationKernel::kernel_start,
                             -highest_mesh_index_to_particle[0])
                  : InterpolationKernel::kernel_start,
              is_out_or_almost_out_of_domain
                  ? std::max(InterpolationKernel::kernel_start,
                             -highest_mesh_index_to_particle[1])
                  : InterpolationKernel::kernel_start};
          const int end[2] = {
              is_out_or_almost_out_of_domain
                  ? std::min(InterpolationKernel::kernel_end,
                             output_field_at_mesh_size1 -
                                 highest_mesh_index_to_particle[0])
                  : InterpolationKernel::kernel_end,
              is_out_or_almost_out_of_domain
                  ? std::min(InterpolationKernel::kernel_end,
                             output_field_at_mesh_size0 -
                                 highest_mesh_index_to_particle[1])
                  : InterpolationKernel::kernel_end};

          for (int sy = start[1]; sy < end[1]; ++sy) {
            const int scaled_y_mesh_index =
                output_field_at_mesh_size1 *
                (highest_mesh_index_to_particle[1] + sy);
            const Float y_weight =
                weights[sy - InterpolationKernel::kernel_start][1];
            for (int sx = start[0]; sx < end[0]; ++sx) {
              const int mesh_index =
                  // This is the x_mesh index
                  (highest_mesh_index_to_particle[0] + sx) +
                  //
                  scaled_y_mesh_index;
              const Float weight =
                  y_weight * weights[sx - InterpolationKernel::kernel_start][0];
              output_field_at_mesh[mesh_index] += weight * input;
            }  // sx
          }    // sy
        }      // Output scope
      }        // i_particle
    }          // j_particle
  }

}  // namespace detail
