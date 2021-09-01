#pragma once

// TODO : Type traits for kernels
// include testing the number of f() functions based on the type
// example MP4 has half kernel width 2 and needs f0, f1
#include "particle_kernels/KernelWrapper.hpp"
#include "particle_kernels/LinearKernel.hpp"
#include "particle_kernels/MP4.hpp"
#include "particle_kernels/MP6.hpp"
#include "particle_kernels/QuadraticSpline.hpp"
#include "particle_kernels/YangSmoothThreePointKernel.hpp"
