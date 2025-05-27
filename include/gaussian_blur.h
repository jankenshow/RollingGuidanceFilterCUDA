#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

namespace rgf {
// Gaussian blur (CUDA) interface
// input/output: H x W x C (row-major), float32, C=1 (grayscale) or 3 (color)
void gaussian_blur_cuda(const float* input, float* output, int width,
                        int height, int channels, float sigma);

}  // namespace rgf

#endif  // GAUSSIAN_BLUR_H