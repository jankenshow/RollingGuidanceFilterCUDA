#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

namespace rgf {
__global__ void gaussian_blur_kernel(const float* input, float* output,
                                     int width, int height, float sigma);
__global__ void gaussian_blur_kernel_multi(const float* input, float* output,
                                           int width, int height, int channels,
                                           float sigma);

__global__ void rgf_bilateral_kernel(const float* input, const float* guide,
                                     float* output, int width, int height,
                                     float sigma_s, float sigma_r);
__global__ void rgf_bilateral_kernel_multi(const float* input,
                                           const float* guide, float* output,
                                           int width, int height, int channels,
                                           float sigma_s, float sigma_r);
}  // namespace rgf

#endif  // KERNELS_H
