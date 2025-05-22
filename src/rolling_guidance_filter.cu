#include "gaussian_blur.h"
#include "kernels.h"
#include "utils.h"

namespace rgf {

__global__ void rgf_bilateral_kernel(const float* input, const float* guide, float* output, int width, int height, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    float center = guide[idx];
    float norm = 0.0f, sum = 0.0f;
    int radius = int(ceilf(1.5f * sigma_s));
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = clamp(x + dx, 0, width - 1);
            int yy = clamp(y + dy, 0, height - 1);
            int nidx = yy * width + xx;
            float spatial = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            float range = expf(-((guide[nidx] - center) * (guide[nidx] - center)) / (2.0f * sigma_r * sigma_r));
            float w = spatial * range;
            norm += w;
            sum += w * input[nidx];
        }
    }
    output[idx] = sum / norm;
}


__global__ void rgf_bilateral_kernel_multi(const float* input, const float* guide, float* output, int width, int height, int channels, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int radius = int(ceilf(1.5f * sigma_s));
    for (int c = 0; c < channels; ++c) {
        int idx = (y * width + x) * channels + c;
        float center = guide[idx];
        float norm = 0.0f, sum = 0.0f;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int xx = clamp(x + dx, 0, width - 1);
                int yy = clamp(y + dy, 0, height - 1);
                int nidx = (yy * width + xx) * channels + c;
                float spatial = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
                float range = expf(-((guide[nidx] - center) * (guide[nidx] - center)) / (2.0f * sigma_r * sigma_r));
                float w = spatial * range;
                norm += w;
                sum += w * input[nidx];
            }
        }
        output[idx] = sum / norm;
    }
}


void rolling_guidance_filter_cuda(const float* input, float* output, int width, int height, int channels, float sigma_s, float sigma_r, int iterations) {
    size_t bytes = width * height * channels * sizeof(float);
    float *d_input = nullptr, *d_ping = nullptr, *d_pong = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_ping, bytes);
    cudaMalloc(&d_pong, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ping, input, bytes, cudaMemcpyHostToDevice); // initial guidance = input

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    if (channels == 1) {
        rgf::gaussian_blur_kernel<<<blocks, threads>>>(d_input, d_ping, width, height, sigma_s);
    } else {
        rgf::gaussian_blur_kernel_multi<<<blocks, threads>>>(d_input, d_ping, width, height, channels, sigma_s);
    }
    
    for (int i = 1; i < iterations; ++i) {
        if (channels == 1) {
            rgf_bilateral_kernel<<<blocks, threads>>>(d_input, d_ping, d_pong, width, height, sigma_s, sigma_r);
        } else {
            rgf_bilateral_kernel_multi<<<blocks, threads>>>(d_input, d_ping, d_pong, width, height, channels, sigma_s, sigma_r);
        }
        std::swap(d_ping, d_pong);
    }
    
    cudaMemcpy(output, d_ping, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_ping);
    cudaFree(d_pong);
}
} // namespace rgf
