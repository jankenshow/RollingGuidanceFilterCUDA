#include "gaussian_blur.h"
#include "kernels.h"
#include "utils.h"
#include <iostream>

namespace rgf {

__global__ void gaussian_blur_kernel(cudaTextureObject_t texInput, cudaSurfaceObject_t surfOutput, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int radius = int(ceilf(2.0f * sigma));
    float sum = 0.0f;
    float norm = 0.0f;
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
            
            float weight = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            sum += weight * uchar1_to_float(tex2D<uchar1>(texInput, x + dx, y + dy));
            norm += weight;
        }
    }
    
    surf2Dwrite(float_to_uchar1(sum / norm), surfOutput, x * sizeof(uchar1), y);
}

__global__ void gaussian_blur_kernel_multi(cudaTextureObject_t texInput, cudaSurfaceObject_t surfOutput, int width, int height, int channels, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int radius = int(ceilf(2.0f * sigma));

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f), norm = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
            
            float4 weight;
            weight.x = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            weight.y = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            weight.z = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            weight.w = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            sum += weight * uchar4_to_float4(tex2D<uchar4>(texInput, x + dx, y + dy));
            norm += weight;
        }
    }
    
    surf2Dwrite(float4_to_uchar4(sum / norm), surfOutput, x * sizeof(uchar4), y);
}

void gaussian_blur_cuda(const unsigned char* input, unsigned char* output, int width, int height, int channels, float sigma) {
    unsigned char* input_tmp, *output_tmp;
    cudaArray_t d_inputArray, d_outputArray;
    cudaChannelFormatDesc channelDesc;
    if (channels == 1) {
        channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    } else if (channels == 3) {
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        input_tmp = new unsigned char[width * height * 4];
        for (int i = 0; i < width * height; i++) {
            input_tmp[i * 4] = input[i * 3];
            input_tmp[i * 4 + 1] = input[i * 3 + 1];
            input_tmp[i * 4 + 2] = input[i * 3 + 2];
            input_tmp[i * 4 + 3] = 0;
        }
    } else if (channels == 4) {
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    } else {
        std::cout << "Invalid number of channels: %d\n" << std::endl;
        return;
    }
    CHECK(cudaMallocArray(&d_inputArray, &channelDesc, width, height));
    CHECK(cudaMallocArray(&d_outputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));
    
    if (channels == 3) {
        CHECK(cudaMemcpy2DToArray(d_inputArray, 0, 0, input_tmp, width * 4 * sizeof(unsigned char), width * 4 * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    } else {
        CHECK(cudaMemcpy2DToArray(d_inputArray, 0, 0, input, width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    }

    cudaTextureObject_t texInput;
    cudaSurfaceObject_t surfOutput;

    // テクスチャの設定
    cudaResourceDesc texRes = {};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_inputArray;

    cudaTextureDesc texDescr = {};
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
    
    CHECK(cudaCreateTextureObject(&texInput, &texRes, &texDescr, nullptr));

    // サーフェスの設定
    cudaResourceDesc surfRes = {};
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = d_outputArray;
    CHECK(cudaCreateSurfaceObject(&surfOutput, &surfRes));
    
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    
    if (channels == 1) {
        gaussian_blur_kernel<<<blocks, threads>>>(texInput, surfOutput, width, height, sigma);
    } else {
        gaussian_blur_kernel_multi<<<blocks, threads>>>(texInput, surfOutput, width, height, channels, sigma);
    }
    CHECK(cudaDeviceSynchronize());
    
    if (channels == 3) {
        output_tmp = new unsigned char[width * height * 4];
        CHECK(cudaMemcpy2DFromArray(output_tmp, width * 4 * sizeof(unsigned char), d_outputArray, 0, 0, width * 4 * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));
        for (int i = 0; i < width * height; i++) {
            output[i * 3] = output_tmp[i * 4];
            output[i * 3 + 1] = output_tmp[i * 4 + 1];
            output[i * 3 + 2] = output_tmp[i * 4 + 2];
        }
        delete[] input_tmp;
        delete[] output_tmp;
    } else {
        CHECK(cudaMemcpy2DFromArray(output, width * channels * sizeof(unsigned char), d_outputArray, 0, 0, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));
    }
    // リソースの解放
    CHECK(cudaDestroyTextureObject(texInput));
    CHECK(cudaDestroySurfaceObject(surfOutput));
    CHECK(cudaFreeArray(d_inputArray));
    CHECK(cudaFreeArray(d_outputArray));
}

} // namespace rgf 