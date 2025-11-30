#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>
#include <stdint.h>

#include <oqmc/sobolbn.h>

#include "math/constant.h"
#include "math/matrix.h"
#include "math/vector.h"

#include "bsdf.h"
#include "camera.h"
#include "image.h"
#include "params.h"
#include "shading_vertex.h"
#include "trace.h"

// Raygen: レイを飛ばす
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int image_index = idx.y * dim.x + idx.x;

    enum DomainKey {
        Next,
    };

    char* sobol_data = reinterpret_cast<char*>(params.sobol_data);
    oqmc::SobolBnSampler sampler(idx.x, idx.y, 0, params.N, sobol_data);

    float u1[1];
    float u2[2];
    sampler.drawSample<2>(u2);

    float3 ray_origin;
    float3 ray_direction;
    generate_camera_ray(make_uint2(idx.x, idx.y), make_uint2(dim.x, dim.y), ray_origin, ray_direction, u2);

    HitInfo hit_info = trace_ray(ray_origin, ray_direction);

    float z = hit_info.t / T_MAX;
    accumulate_depth_pixel(image_index, z);

    if (!hit_info.hit) {
        float4 result;
        if (params.film_transparent) {
            result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            result = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        accumulate_color_pixel(image_index, result);
        return;
    }

    ShadingVertex vtx = get_shading_vertex(hit_info);
    Bsdf bsdf = get_bsdf(hit_info, vtx);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    radiance += throughput * bsdf.emission;

    uint32_t depth = 0;
    while (true) {
        float3 wi;
        float3 weight;
        float pdf;
        SampleType sample_type;

        sampler = sampler.newDomain(Next);
        sampler.drawSample<1>(u1);
        sampler = sampler.newDomain(Next);
        sampler.drawSample<2>(u2);

        float3 wo = normalize(vtx.tangent_frame * (-ray_direction));

        if (!bsdf.sample(wo, u1[0], u2, wi, weight, pdf, sample_type)) {
            break;
        }

        ray_direction = normalize(inverse(vtx.tangent_frame) * wi);
        ray_origin = vtx.position + ray_direction * 1e-8f;

        throughput *= weight / pdf;

        hit_info = trace_ray(ray_origin, ray_direction);
        if (!hit_info.hit) {
            break;
        }

        vtx = get_shading_vertex(hit_info);
        bsdf = get_bsdf(hit_info, vtx);

        radiance += throughput * bsdf.emission;

        if (depth >= 5) {
            sampler = sampler.newDomain(Next);
            sampler.drawSample<1>(u1);

            float roulette_prob = max(throughput.x, max(throughput.y, throughput.z));
            if (u1[0] >= roulette_prob) {
                break;
            }
            throughput /= roulette_prob;
        }

        depth++;
        if (depth >= params.max_depth) {
            break;
        }
    }

    float4 result = make_float4(radiance.x, radiance.y, radiance.z, 1.0f);
    result.x *= params.exposure;
    result.y *= params.exposure;
    result.z *= params.exposure;

    accumulate_color_pixel(image_index, result);
}
