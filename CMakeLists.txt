cmake_minimum_required(VERSION 3.18)
project(rolling_guidance_filter LANGUAGES CXX CUDA)

# CUDA settings
find_package(CUDA REQUIRED)

# pybind11 (for Python bindings)
find_package(pybind11 REQUIRED)

include_directories(include)

add_library(rgf SHARED
    src/rolling_guidance_filter.cu
    # src/rolling_guidance_filter.cpp
    src/gaussian_blur.cu
    src/utils.cu
)
set_target_properties(rgf PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rgf
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(rgf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Python binding
pybind11_add_module(rgf_pybind python/rolling_guidance_filter_pybind.cpp)
target_link_libraries(rgf_pybind PRIVATE rgf)

# Build instructions (example):
# mkdir build && cd build
# cmake .. -DPYBIND11_DIR=$(python3 -m pybind11 --cmakedir)
# make
# (for Python: export PYTHONPATH=$PYTHONPATH:$(pwd))
