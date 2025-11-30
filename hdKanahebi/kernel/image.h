#pragma once

#include "math/vector.h"

#include "params.h"

__device__ void accumulate_color_pixel(const uint32_t pixel_index, const float4& color) {
    float4* color_image = reinterpret_cast<float4*>(params.color_image);

    if (params.N == 1) {
        color_image[pixel_index] = make_float4(color.x, color.y, color.z, 1.0f);
        return;
    } else {
        float N = static_cast<float>(params.N);

        float4 prev_color = color_image[pixel_index];
        float4 new_color;
        new_color = prev_color * N / (N + 1.0f) + color / (N + 1.0f);
        color_image[pixel_index] = new_color;
    }
}

__device__ void accumulate_depth_pixel(const uint32_t pixel_index, const float& depth) {
    float* depth_image = reinterpret_cast<float*>(params.depth_image);

    if (params.N == 1) {
        depth_image[pixel_index] = depth;
        return;
    } else {
        float N = static_cast<float>(params.N);

        float prev_depth = depth_image[pixel_index];
        float new_depth = prev_depth * N / (N + 1.0f) + depth / (N + 1.0f);
        depth_image[pixel_index] = new_depth;
    }
}
