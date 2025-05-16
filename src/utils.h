#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <math.h>

namespace rgf {

__host__ __device__ int clamp(int v, int low, int high);

} // namespace rgf

#endif // UTILS_H