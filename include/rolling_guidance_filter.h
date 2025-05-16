#pragma once
#include <cstddef>

namespace rgf {
// Rolling Guidance Filter (CUDA) interface
// input/output: H x W x C (row-major), float32, C=1 (grayscale) or 3 (color) or 4 (RGBA)
void rolling_guidance_filter_cuda(const unsigned char* input, unsigned char* output, int width,
                                  int height, int channels, float sigma_s,
                                  float sigma_r, int iterations);
}  // namespace rgf
