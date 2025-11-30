#pragma once

#include "math/matrix.h"
#include "math/vector.h"

#include "generalized_schlick.h"
#include "lambert.h"
#include "params.h"
#include "shading_vertex.h"

struct Bsdf {
    float3 albedo;
    float metallic;
    float roughness;
    float3 emission;
    float3 normal;
    float opacity;
    float ior;

private:
    __device__ bool sample_metallic(const float3& wo,
                                    float uc,
                                    float uv[2],
                                    float3& wi,
                                    float3& weight,
                                    float& pdf,
                                    SampleType& sample_type) const {
        float alpha = roughness * roughness;
        float3 r0 = albedo;
        float3 r90 = make_float3(1.0f);
        float3 tint = make_float3(1.0f);
        GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0f, tint, ScatterMode::R,
                                       1.0f,   // etaは反射では使用されない
                                       true,   // entering
                                       false,  // thin_surface
                                       alpha, alpha);
        return gs_bsdf.sample(wo, uv, wi, weight, pdf, sample_type);
    }

    __device__ bool sample_dielectric(const float3& wo,
                                      float uc,
                                      float uv[2],
                                      float3& wi,
                                      float3& weight,
                                      float& pdf,
                                      SampleType& sample_type) const {
        float alpha = roughness * roughness;

        if (uc > opacity) {
            // 透明
            uc = (uc - opacity) / (1.0f - opacity);

            float3 r0 = make_float3(((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f)));
            float3 r90 = make_float3(1.0f);
            float3 tint = make_float3(1.0f);
            bool entering = wo.z > 0.0f;
            GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0, tint, ScatterMode::RT, ior, entering,
                                           false,  // thin_surface
                                           alpha, alpha);

            return gs_bsdf.sample(wo, uv, wi, weight, pdf, sample_type);
        } else {
            // 不透明
            uc = uc / opacity;

            float3 r0 = make_float3(((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f)));
            float3 r90 = make_float3(1.0f);
            float3 tint = make_float3(1.0f);
            GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0f, tint, ScatterMode::R, ior, true,
                                           false,  // thin_surface
                                           alpha, alpha);
            float fresnel = gs_bsdf.fresnel(wo);

            if (uc < fresnel) {
                // 反射成分をサンプリング
                bool result = gs_bsdf.sample(wo, uv, wi, weight, pdf, sample_type);
                pdf *= fresnel;
                return result;
            } else {
                // 拡散成分をサンプリング
                NormalizedLambert lambert_bsdf(albedo * (1.0f - fresnel));
                bool result = lambert_bsdf.sample(wo, uv, wi, weight, pdf, sample_type);
                pdf *= (1.0f - fresnel);
                return result;
            }
        }
    }

    __device__ float3 eval_metallic(const float3& wo, const float3& wi) const {
        float alpha = roughness * roughness;
        float3 r0 = albedo;
        float3 r90 = make_float3(1.0f);
        float3 tint = make_float3(1.0f);
        GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0f, tint, ScatterMode::R,
                                       1.0f,   // etaは反射では使用されない
                                       true,   // entering
                                       false,  // thin_surface
                                       alpha, alpha);
        return gs_bsdf.eval(wo, wi);
    }

    __device__ float3 eval_dielectric(const float3& wo, const float3& wi) const {
        float alpha = roughness * roughness;

        float3 r0 = make_float3(((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f)));
        float3 r90 = make_float3(1.0f);
        float3 tint = make_float3(1.0f);
        bool entering = wo.z > 0.0f;

        // 透明
        GeneralizedSchlickBsdf transparent_bsdf(r0, r90, 5.0, tint, ScatterMode::RT, ior, entering,
                                                false,  // thin_surface
                                                alpha, alpha);
        float3 transparent = transparent_bsdf.eval(wo, wi);

        // 不透明
        GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0, tint, ScatterMode::R, ior, entering,
                                       false,  // thin_surface
                                       alpha, alpha);
        float fresnel = gs_bsdf.fresnel(wo);

        float3 opaque_specular = gs_bsdf.eval(wo, wi);
        NormalizedLambert lambert_bsdf(albedo);
        float3 opaque_diffuse = lambert_bsdf.eval(wo, wi);

        return transparent * opacity + (opaque_specular + opaque_diffuse * (1.0f - fresnel)) * (1.0f - opacity);
    }

    __device__ float pdf_metallic(const float3& wo, const float3& wi) const {
        float alpha = roughness * roughness;
        float3 r0 = albedo;
        float3 r90 = make_float3(1.0f);
        float3 tint = make_float3(1.0f);
        GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0f, tint, ScatterMode::R,
                                       1.0f,   // etaは反射では使用されない
                                       true,   // entering
                                       false,  // thin_surface
                                       alpha, alpha);
        return gs_bsdf.pdf(wo, wi);
    }

    __device__ float pdf_dielectric(const float3& wo, const float3& wi) const {
        float alpha = roughness * roughness;

        float3 r0 = make_float3(((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f)));
        float3 r90 = make_float3(1.0f);
        float3 tint = make_float3(1.0f);
        bool entering = wo.z > 0.0f;

        // 透明
        GeneralizedSchlickBsdf transparent_bsdf(r0, r90, 5.0, tint, ScatterMode::RT, ior, entering,
                                                false,  // thin_surface
                                                alpha, alpha);
        float transparent_pdf = transparent_bsdf.pdf(wo, wi);

        // 不透明
        GeneralizedSchlickBsdf gs_bsdf(r0, r90, 5.0, tint, ScatterMode::R, ior, entering,
                                       false,  // thin_surface
                                       alpha, alpha);
        float fresnel = gs_bsdf.fresnel(wo);
        float opaque_pdf = gs_bsdf.pdf(wo, wi);

        NormalizedLambert lambert_bsdf(albedo);
        float diffuse_pdf = lambert_bsdf.pdf(wo, wi);

        return transparent_pdf * opacity + (opaque_pdf * fresnel + diffuse_pdf * (1.0f - fresnel)) * (1.0f - opacity);
    }

public:
    __device__ bool sample(const float3& wo,
                           float uc,
                           float uv[2],
                           float3& wi,
                           float3& weight,
                           float& pdf,
                           SampleType& sample_type) const {
        if (metallic >= 1.0) {
            // 完全鏡面
            return sample_metallic(wo, uc, uv, wi, weight, pdf, sample_type);
        } else if (metallic <= 0.0) {
            // 完全非金属
            return sample_dielectric(wo, uc, uv, wi, weight, pdf, sample_type);
        } else {
            // 金属と非金属の混合
            if (uc < metallic) {
                // 鏡面成分をサンプリング
                uc /= metallic;
                return sample_metallic(wo, uc, uv, wi, weight, pdf, sample_type);
            } else {
                // 非金属成分をサンプリング
                uc = (uc - metallic) / (1.0f - metallic);
                return sample_dielectric(wo, uc, uv, wi, weight, pdf, sample_type);
            }
        }
    }

    __device__ float3 eval(const float3& wo, const float3& wi) const {
        if (metallic >= 1.0) {
            // 完全鏡面
            return eval_metallic(wo, wi);
        } else if (metallic <= 0.0) {
            // 完全非金属
            return eval_dielectric(wo, wi);
        } else {
            // 金属と非金属の混合
            float3 metallic_eval = eval_metallic(wo, wi);
            float3 dielectric_eval = eval_dielectric(wo, wi);
            return metallic_eval * metallic + dielectric_eval * (1.0f - metallic);
        }
    }

    __device__ float pdf(const float3& wo, const float3& wi) const {
        if (metallic >= 1.0) {
            // 完全鏡面
            return pdf_metallic(wo, wi);
        } else if (metallic <= 0.0) {
            // 完全非金属
            return pdf_dielectric(wo, wi);
        } else {
            // 金属と非金属の混合
            float metallic_pdf = pdf_metallic(wo, wi);
            float dielectric_pdf = pdf_dielectric(wo, wi);
            return metallic_pdf * metallic + dielectric_pdf * (1.0f - metallic);
        }
    }
};

__device__ Bsdf get_bsdf(const HitInfo& hit_info, const ShadingVertex& vtx) {
    SceneParams* scene_params = reinterpret_cast<SceneParams*>(params.scene_params);
    MaterialParams* materials = reinterpret_cast<MaterialParams*>(scene_params->materials);
    MaterialParams& mat = materials[hit_info.material_index];

    Bsdf bsdf;
    bsdf.albedo = mat.base_color;
    bsdf.metallic = mat.metallic;
    bsdf.roughness = mat.roughness;
    bsdf.emission = mat.emissive_color;
    bsdf.normal = make_float3(0.0f, 0.0f, 1.0f);
    bsdf.opacity = mat.opacity;
    bsdf.ior = mat.ior;

    if (mat.has_base_color_texture) {
        float4 albedo = tex2D<float4>(mat.base_color_texture, vtx.uv.x, vtx.uv.y);
        bsdf.albedo = make_float3(albedo.x, albedo.y, albedo.z);
        bsdf.albedo = powf(bsdf.albedo, 2.2f);  // ガンマ補正解除
    }

    if (mat.has_normal_texture) {
        float4 normal_tex = tex2D<float4>(mat.normal_texture, vtx.uv.x, vtx.uv.y);
        float3 normal = make_float3(normal_tex.x * 2.0f - 1.0f, normal_tex.y * 2.0f - 1.0f, normal_tex.z * 2.0f - 1.0f);
        bsdf.normal = normalize(normal);
    }

    if (mat.has_metallic_texture) {
        float4 metallic = tex2D<float4>(mat.metallic_texture, vtx.uv.x, vtx.uv.y);
        bsdf.metallic *= metallic.x;
    }

    if (mat.has_roughness_texture) {
        float4 roughness = tex2D<float4>(mat.roughness_texture, vtx.uv.x, vtx.uv.y);
        bsdf.roughness = roughness.x;
    }

    if (mat.has_emissive_color_texture) {
        float4 emissive = tex2D<float4>(mat.emissive_color_texture, vtx.uv.x, vtx.uv.y);
        bsdf.emission.x *= emissive.x;
        bsdf.emission.y *= emissive.y;
        bsdf.emission.z *= emissive.z;
        bsdf.emission = powf(bsdf.emission, 2.2f);  // ガンマ補正解除
    }

    return bsdf;
}
