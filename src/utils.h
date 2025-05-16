#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

namespace rgf {
  __device__ inline int clamp(int v, int low, int high);
  inline __host__ __device__ void operator+=(float4& a, float4 b);
  inline __host__ __device__ void operator-=(float4& a, float4 b);
  inline __host__ __device__ float4 operator-(float4 a);
  inline __host__ __device__ float4 operator+(float4 a, float4 b);
  inline __host__ __device__ float4 operator*(float4 a, float4 b);
  inline __host__ __device__ float4 operator/(float a, float4 b);
  __device__ inline float uchar_to_float(uchar1 c);
  __device__ inline uchar1 float_to_uchar1(float f);
  __device__ inline float4 uchar4_to_float4(uchar4 c);
  __device__ inline uchar4 float4_to_uchar4(float4 c);

} // namespace rgf
#endif // UTILS_H