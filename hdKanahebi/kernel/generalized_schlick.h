#pragma once

#include "math/matrix.h"
#include "math/vector.h"

#include "enum.h"
#include "params.h"
#include "shading_vertex.h"

struct GeneralizedSchlickBsdf {
    float3 r0;
    float3 r90;
    float exponent;
    float3 tint;
    ScatterMode scatter_mode;
    float eta;
    bool entering;
    bool thin_surface;
    float alpha_x;
    float alpha_y;

    __device__ GeneralizedSchlickBsdf(const float3& _r0,
                                      const float3& _r90,
                                      float _exp,
                                      const float3& _tint,
                                      ScatterMode _mode,
                                      const float& _eta,
                                      bool _entering,
                                      bool _thin,
                                      float _ax,
                                      float _ay)
        : r0(_r0),
          r90(_r90),
          exponent(_exp),
          tint(_tint),
          scatter_mode(_mode),
          eta(_eta),
          entering(_entering),
          thin_surface(_thin),
          alpha_x(_ax),
          alpha_y(_ay) {}

private:
    __device__ static inline float cos_theta(const float3& w) { return w.z; }
    __device__ static inline float cos2_theta(const float3& w) { return w.z * w.z; }
    __device__ static inline float abs_cos_theta(const float3& w) { return fabsf(w.z); }

    __device__ static inline float tan2_theta(const float3& w) {
        float c2 = cos2_theta(w);
        return (c2 == 0.f) ? INFINITY : fmaxf(0.f, (1.f - c2) / c2);
    }

    __device__ static inline float cos_phi(const float3& w) {
        float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos2_theta(w)));
        if (sin_theta == 0.f)
            return 1.f;
        return fminf(fmaxf(w.x / sin_theta, -1.f), 1.f);
    }

    __device__ static inline float sin_phi(const float3& w) {
        float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos2_theta(w)));
        if (sin_theta == 0.f)
            return 0.f;
        return fminf(fmaxf(w.y / sin_theta, -1.f), 1.f);
    }

    __device__ static inline float3 reflect(const float3& wo, const float3& n) {
        float d = dot(wo, n);
        return make_float3(2.f * d * n.x - wo.x, 2.f * d * n.y - wo.y, 2.f * d * n.z - wo.z);
    }

    __device__ static inline bool same_hemisphere(const float3& w1, const float3& w2) { return w1.z * w2.z > 0.f; }

    __device__ static inline float2 sample_uniform_disk_polar(const float2& u) {
        float r = sqrtf(u.x);
        float theta = 2.f * PI * u.y;
        return make_float2(r * cosf(theta), r * sinf(theta));
    }

    __device__ static inline bool refract(const float3& wi, const float3& n, float eta, float3& wt) {
        float cosi = dot(wi, n);
        float sin2_i = fmaxf(0.f, 1.f - cosi * cosi);
        float sin2_t = sin2_i / (eta * eta);
        if (sin2_t >= 1.f)
            return false;
        float cost = sqrtf(fmaxf(0.f, 1.f - sin2_t));
        wt = make_float3((-wi.x / eta) + n.x * (cosi / eta - cost), (-wi.y / eta) + n.y * (cosi / eta - cost),
                         (-wi.z / eta) + n.z * (cosi / eta - cost));
        float len2 = dot(wt, wt);
        if (len2 < 1e-12f)
            return false;
        float inv_len = rsqrtf(len2);
        wt.x *= inv_len;
        wt.y *= inv_len;
        wt.z *= inv_len;
        return true;
    }

    __device__ static inline float sq(float x) { return x * x; }
    __device__ static inline float3 sq3(const float3& v) { return make_float3(v.x * v.x, v.y * v.y, v.z * v.z); }

    __device__ bool effectively_smooth() const { return fmaxf(alpha_x, alpha_y) < 1e-3f; }

    __device__ float3 generalized_schlick_fresnel(float cos_theta) const {
        cos_theta = fminf(fmaxf(cos_theta, 0.f), 1.f);
        float one_minus = 1.f - cos_theta;
        const float cmax = 1.f / 7.f;
        const float omcmax = 1.f - cmax;
        float3 base = r0 + (r90 - r0) * powf(one_minus, exponent);
        float3 fmax = r0 + (r90 - r0) * powf(omcmax, exponent);
        float3 a = fmax * (1.f - tint) / (cmax * powf(omcmax, 6.f));
        return base - a * cos_theta * powf(one_minus, 6.f);
    }

    __device__ float microfacet_distribution(const float3& wm) const {
        float t2 = tan2_theta(wm);
        if (!isfinite(t2))
            return 0.f;
        float c4 = sq(cos2_theta(wm));
        float e = t2 * (sq(cos_phi(wm)) / sq(alpha_x) + sq(sin_phi(wm)) / sq(alpha_y));
        return 1.f / (PI * alpha_x * alpha_y * c4 * sq(1.f + e));
    }

    __device__ float lambda(const float3& w) const {
        float t2 = tan2_theta(w);
        if (isinf(t2))
            return 0.f;
        float a2 = sq(cos_phi(w) * alpha_x) + sq(sin_phi(w) * alpha_y);
        return (sqrtf(1.f + a2 * t2) - 1.f) * 0.5f;
    }

    __device__ float masking_g1(const float3& w) const { return 1.f / (1.f + lambda(w)); }
    __device__ float masking_shadowing_g(const float3& wo, const float3& wi) const {
        return 1.f / (1.f + lambda(wo) + lambda(wi));
    }

    __device__ float visible_normal_distribution(const float3& w, const float3& wm) const {
        float c = fabsf(w.z);
        if (c == 0.f)
            return 0.f;
        return masking_g1(w) / c * microfacet_distribution(wm) * fabsf(dot(w, wm));
    }

    __device__ float3 sample_visible_normal(const float3& w, const float2& u) const {
        float3 wh = normalize(make_float3(alpha_x * w.x, alpha_y * w.y, w.z));
        if (wh.z < 0.f)
            wh *= -1.f;
        float3 t1 = (wh.z < 0.99999f) ? normalize(cross(make_float3(0, 0, 1), wh)) : make_float3(1, 0, 0);
        float3 t2 = cross(wh, t1);
        float2 p = sample_uniform_disk_polar(u);
        float h = sqrtf(fmaxf(0.f, 1.f - p.x * p.x));
        float lerp = 0.5f * (1.f + wh.z);
        float p_y = h * (1.f - lerp) + p.y * lerp;
        float pz = sqrtf(fmaxf(0.f, 1.f - p.x * p.x - p_y * p_y));
        float3 nh = t1 * p.x + t2 * p_y + wh * pz;
        float3 r = make_float3(alpha_x * nh.x, alpha_y * nh.y, fmaxf(1e-6f, nh.z));
        return normalize(r);
    }

    __device__ float fresnel_dielectric_scalar(float cosi, float eta) const {
        cosi = fminf(fmaxf(cosi, -1.f), 1.f);
        float ei = 1.f, et = eta;
        float ci = fabsf(cosi);
        float sint = ei / et * sqrtf(fmaxf(0.f, 1.f - ci * ci));
        if (sint >= 1.f)
            return 1.f;
        float ct = sqrtf(fmaxf(0.f, 1.f - sint * sint));
        float rs = (ei * ci - et * ct) / (ei * ci + et * ct);
        float rp = (et * ci - ei * ct) / (et * ci + ei * ct);
        return 0.5f * (rs * rs + rp * rp);
    }

    __device__ float3 fresnel_dielectric_rgb(float cosi, float eta_scalar) const {
        float F = fresnel_dielectric_scalar(cosi, eta_scalar);
        return make_float3(F);
    }

    __device__ bool compute_generalized_half_vector(const float3& wo,
                                                    const float3& wi,
                                                    float eta,
                                                    float3& wm_out) const {
        float cto = wo.z, cti = wi.z;
        bool reflect = (cti * cto > 0.f);
        float etap = reflect ? 1.f : ((cto > 0.f) ? eta : (1.f / eta));
        float3 wm = wi * etap + wo;
        float l2 = dot(wm, wm);
        if (cti == 0.f || cto == 0.f || l2 == 0.f)
            return false;
        wm = normalize(wm);
        if (wm.z < 0.f)
            wm *= -1.f;
        if (dot(wm, wi) * cti < 0.f || dot(wm, wo) * cto < 0.f)
            return false;
        wm_out = wm;
        return true;
    }

    __device__ bool sample_specular(const float3& wo,
                                    const float u[2],
                                    float3& wi,
                                    float3& weight,
                                    float& pdf_val,
                                    SampleType& st) const {
        float3 F = generalized_schlick_fresnel(fabsf(wo.z));
        if (scatter_mode == ScatterMode::R) {
            wi = make_float3(-wo.x, -wo.y, wo.z);
            if (wi.z == 0.f)
                return false;
            weight = F;
            pdf_val = 1.f;
            st = SampleType::SpecularReflection;
            return true;
        }
        if (scatter_mode == ScatterMode::T) {
            if (thin_surface) {
                wi = make_float3(-wo.x, -wo.y, -wo.z);
                if (wi.z == 0.f)
                    return false;
                weight = make_float3(1.f) - F;
                pdf_val = 1.f;
                st = SampleType::SpecularTransmission;
                return true;
            } else {
                float eta_s = entering ? eta : 1.f / eta;
                float3 n = make_float3(0, 0, entering ? 1.f : -1.f);
                if (!refract(wo, n, eta_s, wi))
                    return false;
                float3 Td = make_float3(1.f) - fresnel_dielectric_rgb(fabsf(wo.z), eta_s);
                weight = Td / (eta_s * eta_s);
                pdf_val = 1.f;
                st = SampleType::SpecularTransmission;
                return true;
            }
        }
        if (scatter_mode == ScatterMode::RT) {
            float pr = (F.x + F.y + F.z) * (1.f / 3.f);
            float pt = 1.f - pr;
            if (u[0] < pr / (pr + pt)) {
                wi = make_float3(-wo.x, -wo.y, wo.z);
                if (wi.z == 0.f)
                    return false;
                weight = F;
                pdf_val = pr / (pr + pt);
                st = SampleType::SpecularReflection;
                return true;
            } else {
                if (thin_surface) {
                    wi = make_float3(-wo.x, -wo.y, -wo.z);
                    if (wi.z == 0.f)
                        return false;
                    weight = make_float3(1.f) - F;
                    pdf_val = pt / (pr + pt);
                    st = SampleType::SpecularTransmission;
                    return true;
                } else {
                    float eta_s = entering ? eta : 1.f / eta;
                    float3 n = make_float3(0, 0, entering ? 1.f : -1.f);
                    if (!refract(wo, n, eta_s, wi))
                        return false;
                    float3 Td = (make_float3(1.f) - fresnel_dielectric_rgb(fabsf(wo.z), eta_s));
                    weight = Td / (eta_s * eta_s);
                    pdf_val = pt / (pr + pt);
                    st = SampleType::SpecularTransmission;
                    return true;
                }
            }
        }
        return false;
    }

    __device__ bool sample_microfacet_reflection(const float3& wo,
                                                 const float3& wm,
                                                 const float3& F,
                                                 float prob,
                                                 float3& wi,
                                                 float3& weight,
                                                 float& pdf_val,
                                                 SampleType& st) const {
        wi = reflect(wo, wm);
        if (!same_hemisphere(wo, wi))
            return false;
        float denom = 4.f * fabsf(dot(wo, wm));
        if (denom == 0.f)
            return false;
        float D = microfacet_distribution(wm);
        float G = masking_shadowing_g(wo, wi);
        float cos_o = fabsf(wo.z);
        pdf_val = visible_normal_distribution(wo, wm) / denom * prob;
        weight = F * (D * G / (4.f * cos_o));
        st = SampleType::GlossyReflection;
        return true;
    }

    __device__ bool sample_microfacet_transmission(const float3& wo,
                                                   const float3& wm,
                                                   float prob,
                                                   float3& wi,
                                                   float3& weight,
                                                   float& pdf_val,
                                                   SampleType& st) const {
        if (thin_surface) {
            wi = make_float3(-wo.x, -wo.y, -wo.z);
            if (wi.z == 0.f)
                return false;
            pdf_val = prob;
            weight = make_float3(1.f) - generalized_schlick_fresnel(fabsf(dot(wo, wm)));
            st = SampleType::GlossyTransmission;
            return true;
        }
        float eta_s = entering ? eta : 1.f / eta;
        float3 wm_used = (entering ? wm : -wm);
        if (!refract(wo, wm_used, eta_s, wi))
            return false;
        if (same_hemisphere(wo, wi) || wi.z == 0.f)
            return false;
        float denom = sq(dot(wi, wm) + dot(wo, wm) / eta_s);
        float dwm_dwi = fabsf(dot(wi, wm)) / denom;
        pdf_val = visible_normal_distribution(wo, wm) * dwm_dwi * prob;
        float D = microfacet_distribution(wm);
        float G = masking_shadowing_g(wo, wi);
        float cos_o = fabsf(wo.z);
        float3 T = make_float3(1.f) - generalized_schlick_fresnel(fabsf(dot(wo, wm)));
        weight = T * D * G * fabsf(dot(wi, wm)) * fabsf(dot(wo, wm)) / (denom * cos_o * eta_s * eta_s);
        st = SampleType::GlossyTransmission;
        return true;
    }

    __device__ bool sample_microfacet(const float3& wo,
                                      const float u[2],
                                      float3& wi,
                                      float3& weight,
                                      float& pdf_val,
                                      SampleType& st) const {
        float3 wm = sample_visible_normal(wo, make_float2(u[0], u[1]));
        float3 F = generalized_schlick_fresnel(fabsf(dot(wo, wm)));
        if (scatter_mode == ScatterMode::R) {
            return sample_microfacet_reflection(wo, wm, F, 1.f, wi, weight, pdf_val, st);
        } else if (scatter_mode == ScatterMode::T) {
            return sample_microfacet_transmission(wo, wm, 1.f, wi, weight, pdf_val, st);
        } else {
            float pr = (F.x + F.y + F.z) * (1.f / 3.f);
            float pt = 1.f - pr;
            if (u[0] < pr / (pr + pt))
                return sample_microfacet_reflection(wo, wm, F, pr / (pr + pt), wi, weight, pdf_val, st);
            else
                return sample_microfacet_transmission(wo, wm, pt / (pr + pt), wi, weight, pdf_val, st);
        }
    }

    __device__ float3 evaluate_reflection(const float3& wo, const float3& wi) const {
        if (!same_hemisphere(wo, wi))
            return make_float3(0.f);
        float cos_o = fabsf(wo.z);
        if (cos_o == 0.f)
            return make_float3(0.f);
        float3 wm = normalize(wo + wi);
        float3 F = generalized_schlick_fresnel(fabsf(dot(wo, wm)));
        float D = microfacet_distribution(wm);
        float G = masking_shadowing_g(wo, wi);
        return F * (D * G / (4.f * cos_o));
    }

    __device__ float3 evaluate_transmission(const float3& wo, const float3& wi) const {
        if (thin_surface) {
            if (wi.x != -wo.x || wi.y != -wo.y || wi.z != -wo.z)
                return make_float3(0.f);
            float3 F = generalized_schlick_fresnel(fabsf(wo.z));
            return make_float3(1.f) - F;
        }
        if (same_hemisphere(wo, wi))
            return make_float3(0.f);
        float eta_s = entering ? eta : 1.f / eta;
        float3 wm;
        if (!compute_generalized_half_vector(wo, wi, eta_s, wm))
            return make_float3(0.f);
        float3 Fd = fresnel_dielectric_rgb(fabsf(dot(wo, wm)), eta_s);
        float3 T = make_float3(1.f) - Fd;
        float denom = sq(dot(wi, wm) + dot(wo, wm) / eta_s);
        float D = microfacet_distribution(wm);
        float G = masking_shadowing_g(wo, wi);
        float cos_o = fabsf(wo.z);
        return T * D * G * fabsf(dot(wi, wm)) * fabsf(dot(wo, wm)) / (denom * cos_o * eta_s * eta_s);
    }

    __device__ float3 eval_microfacet(const float3& wo, const float3& wi) const {
        if (scatter_mode == ScatterMode::R) {
            return evaluate_reflection(wo, wi);
        } else if (scatter_mode == ScatterMode::T) {
            return evaluate_transmission(wo, wi);
        } else {
            if (same_hemisphere(wo, wi))
                return evaluate_reflection(wo, wi);
            else
                return evaluate_transmission(wo, wi);
        }
    }

    __device__ float pdf_reflection(const float3& wo, const float3& wi) const {
        if (!same_hemisphere(wo, wi))
            return 0.f;
        float3 wm = normalize(wo + wi);
        float jac = 4.f * fabsf(dot(wo, wm));
        if (jac == 0.f)
            return 0.f;
        return visible_normal_distribution(wo, wm) / jac;
    }

    __device__ float pdf_transmission(const float3& wo, const float3& wi) const {
        if (thin_surface) {
            if (wi.x != -wo.x || wi.y != -wo.y || wi.z != -wo.z)
                return 0.f;
            return 1.f;
        }
        if (same_hemisphere(wo, wi))
            return 0.f;
        float eta_s = entering ? eta : 1.f / eta;
        float3 wm;
        if (!compute_generalized_half_vector(wo, wi, eta_s, wm))
            return 0.f;
        float denom = sq(dot(wi, wm) + dot(wo, wm) / eta_s);
        float dwm_dwi = fabsf(dot(wi, wm)) / denom;
        return visible_normal_distribution(wo, wm) * dwm_dwi;
    }

    __device__ float pdf_microfacet(const float3& wo, const float3& wi) const {
        if (scatter_mode == ScatterMode::R) {
            return pdf_reflection(wo, wi);
        } else if (scatter_mode == ScatterMode::T) {
            return pdf_transmission(wo, wi);
        } else {
            if (same_hemisphere(wo, wi)) {
                float3 wm = normalize(wo + wi);
                float3 F = generalized_schlick_fresnel(fabsf(dot(wo, wm)));
                float pr = (F.x + F.y + F.z) * (1.f / 3.f);
                float pt = 1.f - pr;
                return pdf_reflection(wo, wi) * pr / (pr + pt);
            } else {
                float eta_s = entering ? eta : 1.f / eta;
                float3 wm;
                if (!compute_generalized_half_vector(wo, wi, eta_s, wm))
                    return 0.f;
                float3 F = generalized_schlick_fresnel(fabsf(dot(wo, wm)));
                float pr = (F.x + F.y + F.z) * (1.f / 3.f);
                float pt = 1.f - pr;
                return pdf_transmission(wo, wi) * pt / (pr + pt);
            }
        }
    }

public:
    __device__ bool sample(const float3& wo,
                           const float u[2],
                           float3& wi,
                           float3& weight,
                           float& pdf_val,
                           SampleType& st) const {
        if (wo.z == 0.f)
            return false;
        if (effectively_smooth())
            return sample_specular(wo, u, wi, weight, pdf_val, st);
        else
            return sample_microfacet(wo, u, wi, weight, pdf_val, st);
    }

    __device__ float3 eval(const float3& wo, const float3& wi) const {
        if (effectively_smooth())
            return make_float3(0.f);
        return eval_microfacet(wo, wi);
    }

    __device__ float pdf(const float3& wo, const float3& wi) const {
        if (effectively_smooth())
            return 0.f;
        return pdf_microfacet(wo, wi);
    }

    __device__ float fresnel(const float3& wo) const {
        float cos_theta = fabsf(wo.z);
        float3 fresnel = generalized_schlick_fresnel(cos_theta);
        return (fresnel.x + fresnel.y + fresnel.z) / 3.0f;
    }
};
