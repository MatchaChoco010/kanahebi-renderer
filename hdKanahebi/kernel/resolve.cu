#include <cuda_runtime.h>
#include <stdint.h>

#include "math/vector.h"

extern "C" __global__ void resolve_color_kernel_uint(const float4* color_buffer,
                                                     uchar4* out_color_buffer,
                                                     const uint32_t width,
                                                     const uint32_t height,
                                                     const uint32_t N,
                                                     const bool flip_y) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint32_t index = y * width + x;
    uint32_t output_index = index;
    if (flip_y) {
        output_index = (height - 1 - y) * width + x;
    }

    // Resolve color
    float4 color_value = color_buffer[index];
    float3 color_rgb = make_float3(color_value.x, color_value.y, color_value.z);

    // tone mapping
    color_rgb = color_rgb / (color_rgb + make_float3(1.0f));

    // gamma correction
    color_rgb = powf(color_rgb, 1.0f / 2.2f);

    float4 color = make_float4(color_rgb.x, color_rgb.y, color_rgb.z, color_value.w);

    out_color_buffer[output_index] =
            make_uchar4(static_cast<unsigned char>(color.x * 255), static_cast<unsigned char>(color.y * 255),
                        static_cast<unsigned char>(color.z * 255), static_cast<unsigned char>(color.w * 255));
}

extern "C" __global__ void resolve_color_kernel_float(const float4* color_buffer,
                                                      float4* out_color_buffer,
                                                      const uint32_t width,
                                                      const uint32_t height,
                                                      const uint32_t N,
                                                      const bool flip_y) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint32_t index = y * width + x;
    uint32_t output_index = index;
    if (flip_y) {
        output_index = (height - 1 - y) * width + x;
    }

    // Resolve color
    float4 color_value = color_buffer[index];
    float3 color_rgb = make_float3(color_value.x, color_value.y, color_value.z);

    // tone mapping
    color_rgb = color_rgb / (color_rgb + make_float3(1.0f));

    // gamma correction
    color_rgb = powf(color_rgb, 1.0f / 2.2f);

    float4 color = make_float4(color_rgb.x, color_rgb.y, color_rgb.z, color_value.w);

    out_color_buffer[output_index] = make_float4(color.x, color.y, color.z, color.w);
}

extern "C" __global__ void resolve_depth_kernel(const float* depth_buffer,
                                                float* out_depth_buffer,
                                                const uint32_t width,
                                                const uint32_t height,
                                                const uint32_t N,
                                                const bool flip_y) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint32_t index = y * width + x;
    uint32_t output_index = index;
    if (flip_y) {
        output_index = (height - 1 - y) * width + x;
    }

    // Resolve depth
    float depth = depth_buffer[index];
    out_depth_buffer[output_index] = depth;
}
