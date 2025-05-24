cmake_minimum_required(VERSION 3.18)
project(rolling_guidance_filter LANGUAGES CXX CUDA)

# CUDA settings
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# CUDA optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

include_directories(include)

add_library(rgf SHARED
    src/rolling_guidance_filter.cu
    src/gaussian_blur.cu
    src/utils.cu
    # src/rolling_guidance_filter.cpp
)
set_target_properties(rgf PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(rgf
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Python binding
if(BUILD_PYTHON_BINDING)
    # pybind11 (for Python bindings)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(rgf_pybind python/rolling_guidance_filter_pybind.cpp)
    target_link_libraries(rgf_pybind PRIVATE rgf)
endif()

# Build instructions (example):
# mkdir build && cd build
# cmake .. -DPYBIND11_DIR=$(python3 -m pybind11 --cmakedir)
# make
# (for Python: export PYTHONPATH=$PYTHONPATH:$(pwd))
