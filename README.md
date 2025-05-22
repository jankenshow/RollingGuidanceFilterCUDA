# Rolling Guidance Filter (CUDA/C++/Python)

This project provides a reproduction implementation of the [Rolling Guidance Filter](https://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/) in CUDA, with C++ and Python (pybind11) interfaces.

We are not responsible for any problems or troubles caused by the use of this program.

## Features

- High-performance rolling guidance filter using CUDA
- C++ API
- Python API (via pybind11, numpy array interface)

## Requirements

- Linux
- NVIDIA GPU
- CUDA Toolkit (>= 12.1 is recommended)
- CMake
- Python 3.x (for Python interface)
- pybind11 (for Python interface)
- numpy (for python interface)

## Build Instructions

See `CMakeLists.txt` or `puproject.toml` for more details.  

### For python

```
pip install .
```

### C++ library

```
make build
```


## Usage

### python

```
import numpy as np
import rgf_cuda

gray_img = np.random.rand(100, 100)
color_img = np.random.rand(100, 100, 3)

res_gray = rgf_cuda.rolling_guidance_filter(gray_img, sigma_s=3, sigma_r=10.0, iterations=3)
res_color = rgf_cuda.rolling_guidance_filter(color_img, sigma_s=3, sigma_r=10.0, iterations=3)

print(res_gray.shape)  # (100, 100)
print(res_color.shape)  # (100, 100, 3)
print(res_color.dtype)  # dtype('float32')
```

- 2D or 3D numpy array in np.uint8 or np.float32, np.float64 should work.
- Normalization to [0,1] is not needed, but is recommended.


### C++

under construction.
