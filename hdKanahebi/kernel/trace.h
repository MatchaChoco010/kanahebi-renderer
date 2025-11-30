#pragma once

#include "params.h"

__device__ __constant__ float T_MAX = 1e16f;

struct HitInfo {
    float3 color;
    float t;
    bool hit;
    uint32_t geometry_index;
    uint32_t primitive_index;
    float3 barycentric_coords;
    float4x4 transform;
    uint32_t material_index;
};

template <typename T>
static __forceinline__ __device__ void pack_pointer(T* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T* unpack_pointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    return reinterpret_cast<T*>(uptr);
}

// Miss
extern "C" __global__ void __miss__ms() {
    HitInfo* hit_info = unpack_pointer<HitInfo>(optixGetPayload_0(), optixGetPayload_1());
    hit_info->t = T_MAX;
    hit_info->hit = false;
}

// Closest hit
extern "C" __global__ void __closesthit__ch() {
    float t = optixGetRayTmax();

    HitInfo* hit_info = unpack_pointer<HitInfo>(optixGetPayload_0(), optixGetPayload_1());

    if (params.N == 1) {
        hit_info->color = make_float3(1.0f, 0.0f, 0.0f);
    } else {
        hit_info->color = make_float3(0.0f, 0.0f, 1.0f);
    }
    hit_info->t = t;
    hit_info->hit = true;

    HitGroupData* sbt_record = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    hit_info->geometry_index = sbt_record->geometry_index;

    uint32_t prim_index = optixGetPrimitiveIndex();
    hit_info->primitive_index = static_cast<uint32_t>(prim_index);

    float2 barycentrics = optixGetTriangleBarycentrics();
    hit_info->barycentric_coords = make_float3(1.0f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    hit_info->transform = sbt_record->transform;

    hit_info->material_index = sbt_record->material_index;
}

__device__ HitInfo trace_ray(const float3& ray_origin, const float3& ray_direction) {
    HitInfo hit_info;

    SceneParams* scene_params = reinterpret_cast<SceneParams*>(params.scene_params);

    unsigned int p0 = 0, p1 = 0;
    pack_pointer(&hit_info, p0, p1);
    optixTrace(scene_params->handle, ray_origin, ray_direction,
               1e-3f,  // tmin
               T_MAX,  // tmax
               0.0f,   // rayTime
               0xFF,   // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               p0, p1);

    return hit_info;
}
