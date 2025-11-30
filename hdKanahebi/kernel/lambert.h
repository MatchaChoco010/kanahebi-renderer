#pragma once

#include "math/matrix.h"
#include "math/vector.h"

#include "enum.h"
#include "params.h"
#include "shading_vertex.h"

__device__ float3 sample_cosine_hemisphere(const float u[2]) {
    float r = sqrtf(u[0]);
    float theta = 2.0f * PI * u[1];

    return normalize(make_float3(r * cosf(theta), r * sinf(theta), sqrtf(fmaxf(0.0f, 1.0f - u[0]))));
}

// 正規化Lambert反射
struct NormalizedLambert {
    float3 albedo;

    __device__ NormalizedLambert(const float3& albedo) : albedo(albedo) {}

    __device__ bool sample(const float3& wo,
                           const float u[2],
                           float3& wi,
                           float3& weight,
                           float& pdf,
                           SampleType& sample_type) {
        float wo_cos_n = wo.z;
        if (wo_cos_n == 0.0f) {
            return false;
        }

        // 接空間でのコサイン半球サンプリング
        wi = sample_cosine_hemisphere(u);
        if (wo_cos_n <= 0.0f) {
            wi.z = -wi.z;
        }

        // 接空間でのコサイン項をチェック
        float wi_cos_n = wi.z;
        if (wi_cos_n == 0.0f) {
            return false;
        }

        if (wo_cos_n * wi_cos_n < 0.0f) {
            // 同じ半球上にない場合は無効
            return false;
        }

        // BSDFの値を計算 (cosine termを含む)
        weight = albedo * wi_cos_n / PI;

        // PDFを計算
        pdf = fabsf(wi_cos_n) / PI;

        // サンプル種別を設定
        sample_type = SampleType::DiffuseReflection;

        return true;
    }

    __device__ float3 eval(const float3& wo, const float3& wi) {
        float wo_cos_n = wo.z;
        float wi_cos_n = wi.z;
        if (wo_cos_n == 0.0f || wi_cos_n == 0.0f) {
            return make_float3(0.0f);
        }
        if (wo_cos_n * wi_cos_n <= 0.0f) {
            // 同じ半球上にない場合は無効
            return make_float3(0.0f);
        }

        // BSDFの値を計算 (cosine termを含む)
        return albedo * fabsf(wi_cos_n) / PI;
    }

    __device__ float pdf(const float3& wo, const float3& wi) {
        float wo_cos_n = wo.z;
        float wi_cos_n = wi.z;
        if (wo_cos_n == 0.0f || wi_cos_n == 0.0f) {
            return 0.0f;
        }
        if (wo_cos_n * wi_cos_n <= 0.0f) {
            // 同じ半球上にない場合は無効
            return 0.0f;
        }

        // PDFを計算
        return fabsf(wi_cos_n) / PI;
    }
};
