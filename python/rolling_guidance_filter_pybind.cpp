#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../include/rolling_guidance_filter.h"
#include "../include/gaussian_blur.h"

namespace py = pybind11;

py::array_t<float> rolling_guidance_filter(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    float sigma_s, float sigma_r, int iterations) {
  py::buffer_info buf = input.request();
  int height, width, channels;
  if (buf.ndim == 2) {
    height = buf.shape[0];
    width = buf.shape[1];
    channels = 1;
  } else if (buf.ndim == 3) {
    height = buf.shape[0];
    width = buf.shape[1];
    channels = buf.shape[2];
  } else {
    throw std::runtime_error("Input must be 2D or 3D array (H,W) or (H,W,C)");
  }
  std::vector<ssize_t> out_shape = {height, width};
  if (channels > 1) out_shape.push_back(channels);
  auto output = py::array_t<float>(out_shape);
  rgf::rolling_guidance_filter_cuda(static_cast<const float*>(buf.ptr),
                                    static_cast<float*>(output.request().ptr),
                                    width, height, channels, sigma_s, sigma_r,
                                    iterations);
  return output;
}

py::array_t<float> gaussian_blur(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    float sigma) {
  py::buffer_info buf = input.request();
  int height, width, channels;
  if (buf.ndim == 2) {
    height = buf.shape[0];
    width = buf.shape[1];
    channels = 1;
  } else if (buf.ndim == 3) {
    height = buf.shape[0];
    width = buf.shape[1];
    channels = buf.shape[2];
  } else {
    throw std::runtime_error("Input must be 2D or 3D array (H,W) or (H,W,C)");
  }
  std::vector<ssize_t> out_shape = {height, width};
  if (channels > 1) out_shape.push_back(channels);
  auto output = py::array_t<float>(out_shape);
  rgf::gaussian_blur_cuda(static_cast<const float*>(buf.ptr),
                     static_cast<float*>(output.request().ptr),
                     width, height, channels, sigma);
  return output;
}

PYBIND11_MODULE(rgf_pybind, m) {
  m.doc() = "CUDA Rolling Guidance Filter";
  m.def("rolling_guidance_filter", &rolling_guidance_filter,
        py::arg("input"), py::arg("sigma_s"), py::arg("sigma_r"),
        py::arg("iterations"),
        "Apply rolling guidance filter (CUDA) to a 2D or 3D float32 numpy "
        "array.");
  m.def("gaussian_blur", &gaussian_blur,
        py::arg("input"), py::arg("sigma"),
        "Apply gaussian blur (CUDA) to a 2D or 3D float32 numpy "
        "array.");
}
