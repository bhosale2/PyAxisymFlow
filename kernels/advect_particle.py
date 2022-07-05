import numpy as np
import core.particles_to_mesh as p2m


def advect_vorticity_via_particles(z_particles, r_particles, vort_particles, vorticity, Z_double, R_double, grid_size_r, u_z, u_r, dx, dt):
    """
    advects vorticity using particles
    """
    z_particles[grid_size_r:, :] += u_z * dt
    z_particles[:grid_size_r, :] += np.flip(u_z, axis=0) * dt
    r_particles[grid_size_r:, :] += u_r * dt
    r_particles[:grid_size_r, :] += -np.flip(u_r, axis=0) * dt

    # # remesh
    vort_particles[grid_size_r:, :] = vorticity
    vort_particles[:grid_size_r, :] = -np.flip(vorticity, axis=0)
    vort_double =  0 * Z_double
    p2m.particles_to_mesh_2D_unbounded_mp4(
        z_particles, r_particles, vort_particles, vort_double, dx, dx
    )
    z_particles[...] = Z_double
    r_particles[...] = R_double
    vorticity[...] = vort_double[grid_size_r:, :]


def advect_vorticity_via_particles_periodic(z_particles, r_particles, vort_particles, vorticity, Z_double, R_double, grid_size_r, u_z, u_r, dx, dt):
    """
    advects vorticity using particles
    """
    z_particles[grid_size_r:, :] += u_z * dt
    z_particles[:grid_size_r, :] += np.flip(u_z, axis=0) * dt
    r_particles[grid_size_r:, :] += u_r * dt
    r_particles[:grid_size_r, :] += -np.flip(u_r, axis=0) * dt
    
    # # remesh
    vort_particles[grid_size_r:, :] = vorticity
    vort_particles[:grid_size_r, :] = -np.flip(vorticity, axis=0)
    vort_double =  0 * Z_double
    p2m.particles_to_mesh_2D_mp4(
        z_particles, r_particles, vort_particles, vort_double, dx, dx
    )
    z_particles[...] = Z_double
    r_particles[...] = R_double
    vorticity[...] = vort_double[grid_size_r:, :]
