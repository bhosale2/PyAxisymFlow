import numpy

grid_size_z = 512
domain_AR = 0.5
dx = 1.0 / grid_size_z
grid_size_r = int(domain_AR * grid_size_z)
CFL = 0.1
LCFL = 0.1
eps = numpy.finfo(float).eps

# numba settings
num_threads = 4
fastmath_flag = True
parallel_flag = True
