#include "gaussian_blur.h"
#include "kernels.h"
#include "utils.h"

namespace rgf {

__global__ void rgf_bilateral_kernel(const float* __restrict__ input,
                                     const float* __restrict__ guide,
                                     float* output, int width, int height,
                                     int channels, float sigma_s,
                                     float sigma_r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int radius = int(ceilf(1.5f * sigma_s));

  int idx = (y * width + x) * channels;
  float center[4];
  for (int i = 0; i < channels; ++i) {
    center[i] = guide[idx + i];
  }

  float norm = 0.0f, sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      int xx = clamp(x + dx, 0, width - 1);
      int yy = clamp(y + dy, 0, height - 1);
      int nidx = (yy * width + xx) * channels;

      float spatial = expf(-(dx * dx + dy * dy) / (2.0f * sigma_s * sigma_s));
      float diff_sum = 0.0f;
      for (int i = 0; i < channels; ++i) {
        diff_sum +=
            (guide[nidx + i] - center[i]) * (guide[nidx + i] - center[i]);
      }
      float range = expf(-diff_sum / (2.0f * sigma_r * sigma_r));

      float w = spatial * range;
      norm += w;
      for (int i = 0; i < channels; ++i) {
        sum[i] += w * input[nidx + i];
      }
    }
  }

  for (int i = 0; i < channels; ++i) {
    output[idx + i] = sum[i] / norm;
  }
}

void rolling_guidance_filter_cuda(const float* input, float* output, int width,
                                  int height, int channels, float sigma_s,
                                  float sigma_r, int iterations) {
  size_t bytes = width * height * channels * sizeof(float);
  float *d_input = nullptr, *d_ping = nullptr, *d_pong = nullptr;
  cudaMalloc(&d_input, bytes);
  cudaMalloc(&d_ping, bytes);
  cudaMalloc(&d_pong, bytes);
  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);
  rgf::gaussian_blur_kernel<<<blocks, threads>>>(d_input, d_ping, width, height,
                                                 channels, sigma_s);

  for (int i = 1; i < iterations; ++i) {
    rgf_bilateral_kernel<<<blocks, threads>>>(
        d_input, d_ping, d_pong, width, height, channels, sigma_s, sigma_r);
    std::swap(d_ping, d_pong);
  }

  cudaMemcpy(output, d_ping, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_ping);
  cudaFree(d_pong);
}
}  // namespace rgf
