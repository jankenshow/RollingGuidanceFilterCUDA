#ifndef ROLLING_GUIDANCE_FILTER_H
#define ROLLING_GUIDANCE_FILTER_H

namespace rgf {
// Rolling Guidance Filter (CUDA) interface
// input/output: H x W x C (row-major), float32, C=1 (grayscale) or 3 (color)
void rolling_guidance_filter_cuda(const float* input, float* output, int width,
                                  int height, int channels, float sigma_s,
                                  float sigma_r, int iterations);
}  // namespace rgf

#endif  // ROLLING_GUIDANCE_FILTER_H