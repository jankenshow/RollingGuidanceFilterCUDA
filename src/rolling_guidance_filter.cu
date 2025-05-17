#include "kernels.h"
#include "utils.h"
#include <iostream>

namespace rgf {

__global__ void rgf_bilateral_kernel(cudaTextureObject_t texInput, cudaTextureObject_t texGuide, cudaSurfaceObject_t surfOutput, int width, int height, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    float center = uchar1_to_float(tex2D<uchar1>(texGuide, x, y));
    float norm = 0.0f, sum = 0.0f;
    int radius = int(ceilf(2.0f * sigma_s));
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
                        
            float spatial = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            float guide_val = uchar1_to_float(tex2D<uchar1>(texGuide, x + dx, y + dy));
            float range = expf(-((guide_val - center) * (guide_val - center)) / (2.0f * sigma_r * sigma_r));
            float w = spatial * range;
            
            norm += w;
            sum += w * uchar1_to_float(tex2D<uchar1>(texInput, x + dx, y + dy));
        }
    }
    
    surf2Dwrite(float_to_uchar1(sum / norm), surfOutput, x * sizeof(uchar1), y);
}

__global__ void rgf_bilateral_kernel_multi(cudaTextureObject_t texInput, cudaTextureObject_t texGuide, cudaSurfaceObject_t surfOutput, int width, int height, int channels, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int radius = int(ceilf(2.0f * sigma_s));
    float4 center = uchar4_to_float4(tex2D<uchar4>(texGuide, x, y));
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f), norm = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;

            float4 spatial;
            spatial.x = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            spatial.y = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            spatial.z = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            spatial.w = expf(-(dx*dx + dy*dy) / (2.0f * sigma_s * sigma_s));
            float4 guide_val = uchar4_to_float4(tex2D<uchar4>(texGuide, x + dx, y + dy));
            float4 range;
            range.x = expf(-((guide_val.x - center.x) * (guide_val.x - center.x)) / (2.0f * sigma_r * sigma_r));
            range.y = expf(-((guide_val.y - center.y) * (guide_val.y - center.y)) / (2.0f * sigma_r * sigma_r));
            range.z = expf(-((guide_val.z - center.z) * (guide_val.z - center.z)) / (2.0f * sigma_r * sigma_r));
            float4 w = spatial * range;
            
            norm += w;
            sum += w * uchar4_to_float4(tex2D<uchar4>(texInput, x + dx, y + dy));
        }
    }
    
    surf2Dwrite(float4_to_uchar4(sum / norm), surfOutput, x*sizeof(uchar4), y);
}

void rolling_guidance_filter_cuda(const unsigned char* input, unsigned char* output, int width, int height, int channels, float sigma_s, float sigma_r, int iterations) {
    int orig_channels = channels;
    
    unsigned char *input_tmp, *output_tmp;
    cudaArray_t d_inputArray, d_guideArray, d_outputArray;
    cudaChannelFormatDesc channelDesc;
    if (channels == 1) {
        channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    } else if (channels == 3) {
        channels = 4;
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
    CHECK(cudaMallocArray(&d_guideArray, &channelDesc, width, height));
    CHECK(cudaMallocArray(&d_outputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));
    if (orig_channels == 3) {
        CHECK(cudaMemcpy2DToArray(d_inputArray, 0, 0, input_tmp, width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    } else {
        CHECK(cudaMemcpy2DToArray(d_inputArray, 0, 0, input, width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    }
    
    // テクスチャメモリの宣言
    cudaTextureObject_t texInput;
    cudaTextureObject_t texGuide;
    cudaSurfaceObject_t surfOutput;
    
    // テクスチャの設定
    cudaTextureDesc texDescr = {};
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    cudaResourceDesc texRes = {};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_inputArray;
    CHECK(cudaCreateTextureObject(&texInput, &texRes, &texDescr, nullptr));
    
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_guideArray;
    CHECK(cudaCreateTextureObject(&texGuide, &texRes, &texDescr, nullptr));

    cudaResourceDesc surfRes = {};
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = d_outputArray;
    CHECK(cudaCreateSurfaceObject(&surfOutput, &surfRes));
    
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    
    // ガウシアンブラーを実行
    if (channels == 1) {
        gaussian_blur_kernel<<<blocks, threads>>>(texInput, surfOutput, width, height, sigma_s);
    } else {
        gaussian_blur_kernel_multi<<<blocks, threads>>>(texInput, surfOutput, width, height, channels, sigma_s);
    }
    CHECK(cudaDeviceSynchronize());
    
    // イテレーション実行
    for (int i = 1; i < iterations; ++i) {
        CHECK(cudaMemcpy2DArrayToArray(d_guideArray, 0, 0, d_outputArray, 0, 0, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToDevice));

        if (channels == 1) {
            rgf_bilateral_kernel<<<blocks, threads>>>(texInput, texGuide, surfOutput, width, height, sigma_s, sigma_r);
        } else {
            rgf_bilateral_kernel_multi<<<blocks, threads>>>(texInput, texGuide, surfOutput, width, height, channels, sigma_s, sigma_r);
        }
        CHECK(cudaDeviceSynchronize());
        // std::swap(d_guideArray, d_outputArray);
        // CHECK(cudaCreateTextureObject(&texGuide, &texRes, &texDescr, nullptr));
        // CHECK(cudaCreateSurfaceObject(&surfOutput, &surfRes));
    }
    
    // 結果をホストメモリにコピー
    if (orig_channels == 3) {
        output_tmp = new unsigned char[width * height * channels];
        CHECK(cudaMemcpy2DFromArray(output_tmp, width * channels * sizeof(unsigned char), d_outputArray, 0, 0, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));
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
    CHECK(cudaDestroyTextureObject(texGuide));
    CHECK(cudaDestroySurfaceObject(surfOutput));
    CHECK(cudaFreeArray(d_inputArray));
    CHECK(cudaFreeArray(d_guideArray));
    CHECK(cudaFreeArray(d_outputArray));
}
} // namespace rgf
