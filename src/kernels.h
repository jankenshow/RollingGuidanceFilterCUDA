#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

namespace rgf {
__global__ void gaussian_blur_kernel(cudaTextureObject_t inputTex, cudaSurfaceObject_t outputSurf, int width, int height, float sigma);
__global__ void gaussian_blur_kernel_multi(cudaTextureObject_t inputTex, cudaSurfaceObject_t outputSurf, int width, int height, int channels, float sigma);

__global__ void rgf_bilateral_kernel(cudaTextureObject_t inputTex, cudaTextureObject_t guideTex, cudaSurfaceObject_t outputSurf, int width, int height, float sigma_s, float sigma_r);
__global__ void rgf_bilateral_kernel_multi(cudaTextureObject_t inputTex, cudaTextureObject_t guideTex, cudaSurfaceObject_t outputSurf, int width, int height, int channels, float sigma_s, float sigma_r);
} // namespace rgf

#endif // KERNELS_H
