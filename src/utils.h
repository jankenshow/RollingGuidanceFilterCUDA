#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

namespace rgf {

  __host__ __device__ int clamp(int v, int low, int high);
  __host__ __device__ void operator+=(float4& a, float4 b);
  __host__ __device__ void operator-=(float4& a, float4 b);
  __host__ __device__ float4 operator-(float4 a);
  __host__ __device__ float4 operator+(float4 a, float4 b);
  __host__ __device__ float4 operator*(float4 a, float4 b);
  __host__ __device__ float4 operator/(float4 a, float4 b);
  __host__ __device__ float4 operator*(float4 a, float b);
  __host__ __device__ float4 operator*(float a, float4 b);
  __host__ __device__ float4 operator/(float4 a, float b);
  __host__ __device__ float4 operator/(float a, float4 b);
  __host__ __device__ float uchar_to_float(uchar1 c);
  __host__ __device__ uchar1 float_to_uchar1(float f);
  __host__ __device__ float4 uchar4_to_float4(uchar4 c);
  __host__ __device__ uchar4 float4_to_uchar4(float4 c);

} // namespace rgf

#endif // UTILS_H