#pragma once

#include <cmath>
#include <cstdio>

/*
  2D version of m2p implementation
*/
namespace detail {

  template <class Float, class InterpolationKernel>
  void mesh_to_particles_with_offset_impl_2D(
      const Float input_field_x[], const int input_field_x_size0,
      const int input_field_x_size1, const Float input_field_y[],
      const int input_field_y_size0, const int input_field_y_size1,
      const Float particle_positions_x[], const int particle_positions_x_size0,
      const int particle_positions_x_size1, const Float particle_positions_y[],
      const int particle_positions_y_size0,
      const int particle_positions_y_size1, Float output_field_x[],
      const int output_field_x_size0, const int output_field_x_size1,
      Float output_field_y[], const int output_field_y_size0,
      const int output_field_y_size1, const Float delta_x,
      const Float delta_y) {
    // TODO shouldn't the order be opposite here?
    Float weights[InterpolationKernel::kernel_size][2];
    Float scaled_distance[InterpolationKernel::kernel_size][2];
    Float partial_sums[InterpolationKernel::kernel_size][2];
    Float accumulated[2];

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

        for (int i = 0; i < 2; ++i)
          accumulated[i] = 0.0;

        for (int i = InterpolationKernel::kernel_start;
             i < InterpolationKernel::kernel_end; ++i) {
          for (int j = 0; j < 2; ++j) {
            partial_sums[i - InterpolationKernel::kernel_start][j] = 0.0;
          }
        }

#ifdef INTERNAL_DEBUG_
        printf("Cleared all sums \n");
#endif

        // Do the x interpolation first for all four rows of y
        {
          for (int sy = InterpolationKernel::kernel_start;
               sy < InterpolationKernel::kernel_end; ++sy) {
            const int scaled_y_mesh_index =
                input_field_x_size1 * ((highest_mesh_index_to_particle[1] + sy +
                                        input_field_x_size0) %
                                       input_field_x_size0);
            for (int sx = InterpolationKernel::kernel_start;
                 sx < InterpolationKernel::kernel_end; ++sx) {
              const int mesh_index =
                  // This is the x_mesh index
                  ((highest_mesh_index_to_particle[0] + sx +
                    input_field_x_size1) %
                   input_field_x_size1) +
                  //
                  scaled_y_mesh_index;
              const Float weight =
                  weights[sx - InterpolationKernel::kernel_start][0];

#ifdef INTERNAL_DEBUG_
              if (last_particle) {
                printf("%d, %f\t", mesh_index, weight);
              }
#endif

              // Gathers conttributions from (x-2,x-1,x,x+1) and pools it into
              // (y-2, y-1,y, y+1) located at partial sums
              partial_sums[sy - InterpolationKernel::kernel_start][0] +=
                  weight * input_field_x[mesh_index];
              partial_sums[sy - InterpolationKernel::kernel_start][1] +=
                  weight * input_field_y[mesh_index];
            }
          }
        }
#ifdef INTERNAL_DEBUG_
        printf("x interpolation done \n");
#endif

        // Now do the y interpolation given partial contributions from x
        {
          for (int sy = InterpolationKernel::kernel_start;
               sy < InterpolationKernel::kernel_end; ++sy) {
            const Float weight =
                weights[sy - InterpolationKernel::kernel_start][1];
            for (int i = 0; i < 2; ++i) {
              accumulated[i] +=
                  weight *
                  partial_sums[sy - InterpolationKernel::kernel_start][i];
            }
          }
        }
#ifdef INTERNAL_DEBUG_
        printf("y interpolation done \n");
#endif

        // Put the accumulated quantitiy in the target location
        {
          output_field_x[curr_particle_id] = accumulated[0];
          output_field_y[curr_particle_id] = accumulated[1];
        }

#ifdef INTERNAL_DEBUG_
        if (last_particle)
          printf("\n");
#endif
      }
    }
  }
}  // namespace detail

/*********************************************************************
 *
 *       2D version of m2p implementation
 *       for unbounded cases
 *
 *********************************************************************/
namespace detail {

  template <class Float, class InterpolationKernel>
  void mesh_to_particles_with_offset_impl_2D_unbounded(
      const Float input_field_x[], const int input_field_x_size0,
      const int input_field_x_size1, const Float input_field_y[],
      const int input_field_y_size0, const int input_field_y_size1,
      const Float particle_positions_x[], const int particle_positions_x_size0,
      const int particle_positions_x_size1, const Float particle_positions_y[],
      const int particle_positions_y_size0,
      const int particle_positions_y_size1, Float output_field_x[],
      const int output_field_x_size0, const int output_field_x_size1,
      Float output_field_y[], const int output_field_y_size0,
      const int output_field_y_size1, const Float delta_x,
      const Float delta_y) {
    // TODO shouldn't the order be opposite here?
    Float weights[InterpolationKernel::kernel_size][2];
    Float scaled_distance[InterpolationKernel::kernel_size][2];
    Float partial_sums[InterpolationKernel::kernel_size][2];
    Float accumulated[2];

    // Delegate any checks in the python kernel
    const int stride = particle_positions_x_size1;

// #define INTERNAL_DEBUG_
#ifdef INTERNAL_DEBUG_
    printf("size0 : %d\n", particle_positions_x_size0);
    printf("size1 : %d\n", particle_positions_x_size1);
#endif

    for (int j_particle = 0; j_particle < particle_positions_x_size0;
         ++j_particle) {
      for (int i_particle = 0; i_particle < particle_positions_x_size1;
           ++i_particle) {
#ifdef INTERNAL_DEBUG_
        bool last_particle = (i_particle == (particle_positions_x_size1 - 2)) &&
                             (j_particle == (particle_positions_x_size0 - 2));
#endif

        const int curr_particle_id = i_particle + j_particle * stride;

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("( %d, %d, %d) \n", i_particle, j_particle, curr_particle_id);
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

        for (int i = 0; i < 2; ++i)
          accumulated[i] = 0.0;

        for (int i = InterpolationKernel::kernel_start;
             i < InterpolationKernel::kernel_end; ++i) {
          for (int j = 0; j < 2; ++j) {
            partial_sums[i - InterpolationKernel::kernel_start][j] = 0.0;
          }
        }

#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("Cleared all sums \n");
        }
#endif

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
              InterpolationKernel::kernel_end) > input_field_x_size1) ||
            ((highest_mesh_index_to_particle[1] +
              InterpolationKernel::kernel_end) > input_field_x_size0);

        // Do the x interpolation first for all four rows of y
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
                  ? std::min(
                        InterpolationKernel::kernel_end,
                        input_field_x_size1 - highest_mesh_index_to_particle[0])
                  : InterpolationKernel::kernel_end,
              is_out_or_almost_out_of_domain
                  ? std::min(
                        InterpolationKernel::kernel_end,
                        input_field_x_size0 - highest_mesh_index_to_particle[1])
                  : InterpolationKernel::kernel_end};

#ifdef INTERNAL_DEBUG_
          if (last_particle) {
            printf("Limits of interpolation in x : start = %d , stop = %d\n",
                   start[0], end[0]);
            printf("Limits of interpolation in y : start = %d , stop = %d\n",
                   start[1], end[1]);
          }
#endif

          for (int sy = start[1]; sy < end[1]; ++sy) {
            const int scaled_y_mesh_index =
                input_field_x_size1 * (highest_mesh_index_to_particle[1] + sy);
#ifdef INTERNAL_DEBUG_
            // For formatting the mesh weights below
            if (last_particle) {
              printf("\n");
            }
#endif
            for (int sx = start[0]; sx < end[0]; ++sx) {
              const int mesh_index =
                  // This is the x_mesh index
                  highest_mesh_index_to_particle[0] + sx +
                  //
                  scaled_y_mesh_index;
              const Float weight =
                  weights[sx - InterpolationKernel::kernel_start][0];

#ifdef INTERNAL_DEBUG_
              if (last_particle) {
                printf("%d, %f\t", mesh_index, weight);
              }
#endif

              // Gathers conttributions from (x-2,x-1,x,x+1) and pools it into
              // (y-2, y-1,y, y+1) located at partial sums
              partial_sums[sy - InterpolationKernel::kernel_start][0] +=
                  weight * input_field_x[mesh_index];
              partial_sums[sy - InterpolationKernel::kernel_start][1] +=
                  weight * input_field_y[mesh_index];
            }
          }
        }
#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("\nx interpolation done \n");
        }
#endif

        // Now do the y interpolation given partial contributions from x
        {
          for (int sy = InterpolationKernel::kernel_start;
               sy < InterpolationKernel::kernel_end; ++sy) {
            const Float weight =
                weights[sy - InterpolationKernel::kernel_start][1];
            for (int i = 0; i < 2; ++i) {
              accumulated[i] +=
                  weight *
                  partial_sums[sy - InterpolationKernel::kernel_start][i];
            }
          }
        }
#ifdef INTERNAL_DEBUG_
        if (last_particle) {
          printf("y interpolation done \n");
        }
#endif

        // Put the accumulated quantitiy in the target location
        {
          output_field_x[curr_particle_id] = accumulated[0];
          output_field_y[curr_particle_id] = accumulated[1];
        }

#ifdef INTERNAL_DEBUG_
        if (last_particle)
          printf("\n");
#endif
        // #undef INTERNAL_DEBUG_
      }
    }
  }
}  // namespace detail
