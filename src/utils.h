#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <iostream>
// #include <math.h>

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(call) << std::endl; \
        std::exit(1); \
    }

namespace rgf {

__host__ __device__ int clamp(int v, int low, int high);
__host__ __device__ void operator+=(float4& a, float4 b);
__host__ __device__ void operator-=(float4& a, float4 b);
__host__ __device__ float4 operator-(float4 a);
__host__ __device__ float4 operator+(float4 a, float4 b);
__host__ __device__ float4 operator+(float4 a, float b);
__host__ __device__ float4 operator+(float a, float4 b);
__host__ __device__ float4 operator-(float4 a, float4 b);
__host__ __device__ float4 operator-(float4 a, float b);
__host__ __device__ float4 operator-(float a, float4 b);
__host__ __device__ float4 operator*(float4 a, float4 b);
__host__ __device__ float4 operator*(float4 a, float b);
__host__ __device__ float4 operator*(float a, float4 b);
__host__ __device__ float4 operator/(float4 a, float4 b);
__host__ __device__ float4 operator/(float4 a, float b);
__host__ __device__ float4 operator/(float a, float4 b);
__host__ __device__ float uchar1_to_float(uchar1 c);
__host__ __device__ uchar1 float_to_uchar1(float f);
__host__ __device__ float4 uchar4_to_float4(uchar4 c);
__host__ __device__ uchar4 float4_to_uchar4(float4 c);

} // namespace rgf

// inline __host__ __device__ int clamp(int v, int low, int high) {
//     return max(low, min(v, high));
// }

// inline __host__ __device__ void operator+=(float4& a, float4 b)
// {
//     a.x += b.x;
//     a.y += b.y;
//     a.z += b.z;
//     a.w += b.w;
// }

// inline __host__ __device__ void operator-=(float4& a, float4 b)
// {
//     a.x -= b.x;
//     a.y -= b.y;
//     a.z -= b.z;
//     a.w -= b.w;
// }

// inline __host__ __device__ float4 operator-(float4 a)
// {
//     return make_float4(-a.x, -a.y, -a.z, -a.w);
// }

// inline __host__ __device__ float4 operator+(float4 a, float4 b)
// {
//     return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
// }

// inline __host__ __device__ float4 operator+(float4 a, float b)
// {
//     return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
// }

// inline __host__ __device__ float4 operator+(float a, float4 b)
// {
//     return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
// }


// inline __host__ __device__ float4 operator-(float4 a, float4 b)
// {
//     return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
// }

// inline __host__ __device__ float4 operator-(float4 a, float b)
// {
//     return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
// }

// inline __host__ __device__ float4 operator-(float a, float4 b)
// {
//     return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
// }

// inline __host__ __device__ float4 operator*(float4 a, float4 b)
// {
//     return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
// }

// inline __host__ __device__ float4 operator*(float4 a, float b)
// {
//     return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
// }

// inline __host__ __device__ float4 operator*(float a, float4 b)
// {
//     return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
// }

// inline __host__ __device__ float4 operator/(float4 a, float4 b)
// {
//     return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
// }

// inline __host__ __device__ float4 operator/(float4 a, float b)
// {
//     return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
// }

// inline __host__ __device__ float4 operator/(float a, float4 b)
// {
//     return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
// }

// inline __host__ __device__ float uchar1_to_float(uchar1 c) {
//     return static_cast<float>(c.x) / 255.0f;
// }

// inline __host__ __device__ uchar1 float_to_uchar1(float f) {
//     return make_uchar1(static_cast<unsigned char>(f * 255.0f));
// }

// inline __host__ __device__ float4 uchar4_to_float4(uchar4 c) {
//     return make_float4(static_cast<float>(c.x) / 255.0f, static_cast<float>(c.y) / 255.0f, static_cast<float>(c.z) / 255.0f, static_cast<float>(c.w) / 255.0f);
// }

// inline __host__ __device__ uchar4 float4_to_uchar4(float4 c) {
//     return make_uchar4(static_cast<unsigned char>(c.x * 255.0f), static_cast<unsigned char>(c.y * 255.0f), static_cast<unsigned char>(c.z * 255.0f), static_cast<unsigned char>(c.w * 255.0f));
// }

#endif // UTILS_H