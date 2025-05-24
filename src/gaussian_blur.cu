#include "gaussian_blur.h"
#include "kernels.h"
#include "utils.h"

namespace rgf {

__global__ void gaussian_blur_kernel(const float* input, float* output,
                                     int width, int height, float sigma) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = y * width + x;
  float sum = 0.0f;
  float norm = 0.0f;
  int radius = int(ceilf(2.0f * sigma));

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      int xx = clamp(x + dx, 0, width - 1);
      int yy = clamp(y + dy, 0, height - 1);
      int nidx = yy * width + xx;

      float weight = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
      sum += weight * input[nidx];
      norm += weight;
    }
  }

  output[idx] = sum / norm;
}

__global__ void gaussian_blur_kernel_multi(const float* input, float* output,
                                           int width, int height, int channels,
                                           float sigma) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int radius = int(ceilf(2.0f * sigma));

  for (int c = 0; c < channels; ++c) {
    int idx = (y * width + x) * channels + c;
    float sum = 0.0f;
    float norm = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int xx = clamp(x + dx, 0, width - 1);
        int yy = clamp(y + dy, 0, height - 1);
        int nidx = (yy * width + xx) * channels + c;

        float weight = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
        sum += weight * input[nidx];
        norm += weight;
      }
    }

    output[idx] = sum / norm;
  }
}

void gaussian_blur_cuda(const float* input, float* output, int width,
                        int height, int channels, float sigma) {
  size_t bytes = width * height * channels * sizeof(float);
  float *d_input = nullptr, *d_output = nullptr;

  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_output, bytes);

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  if (channels == 1) {
    gaussian_blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height,
                                              sigma);
  } else {
    gaussian_blur_kernel_multi<<<blocks, threads>>>(d_input, d_output, width,
                                                    height, channels, sigma);
  }

  cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

}  // namespace rgf