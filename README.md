# Rolling Guidance Filter (CUDA/C++/Python)

This project provides a CUDA implementation of the Rolling Guidance Filter, with C++ and Python (pybind11) interfaces.

## Features

- High-performance rolling guidance filter using CUDA
- C++ API
- Python API (via pybind11, numpy array interface)
- Example/test code included

## Requirements

- Linux
- NVIDIA GPU
- CUDA Toolkit (>= 12.1 is recommended)
- CMake
- Python 3.x (for Python interface)
- pybind11 (for Python interface)
- numpy (for python interface)

## Build Instructions

see `CMakeLists.txt` or `puproject.toml` 

### For python

```
pip install .
```

### C++ library

```
make build
```
