#include "rolling_guidance_filter.h"

#include <stdexcept>

namespace rgf {
// C++ wrapper (calls CUDA implementation)
void rolling_guidance_filter(const float* input, float* output, int width,
                             int height, int channels, float sigma_s,
                             float sigma_r, int iterations) {
  rolling_guidance_filter_cuda(input, output, width, height, channels, sigma_s,
                               sigma_r, iterations);
}
}  // namespace rgf
