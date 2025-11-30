#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "vector.h"

#ifndef __DEVICE_BUILD__
#include <glm/glm.hpp>
#endif

struct float3x3 {
    float3 rows[3];

    __host__ __device__ float3x3(float diag = 1.0f) {
        for (int r = 0; r < 3; ++r) {
            rows[r] = make_float3(0.0f);
            ((float*)&rows[r])[r] = diag;
        }
    }

    __host__ __device__ float3x3(const float3& r0, const float3& r1, const float3& r2) {
        rows[0] = r0;
        rows[1] = r1;
        rows[2] = r2;
    }

    __host__ __device__ float* operator[](int row) { return (float*)&rows[row]; }
    __host__ __device__ const float* operator[](int row) const { return (const float*)&rows[row]; }
};

__host__ __device__ inline float3x3 identity3x3() {
    return float3x3(1.0f);
}

__host__ __device__ inline float3x3 transpose(const float3x3& m) {
    float3x3 r(0.0f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ((float*)&r.rows[i])[j] = ((const float*)&m.rows[j])[i];
    return r;
}

__host__ __device__ inline float3x3 operator*(const float3x3& a, const float3x3& b) {
    float3x3 r(0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ((float*)&r.rows[i])[j] = a.rows[i].x * ((const float*)&b.rows[0])[j] +
                                      a.rows[i].y * ((const float*)&b.rows[1])[j] +
                                      a.rows[i].z * ((const float*)&b.rows[2])[j];
        }
    }
    return r;
}

__host__ __device__ inline float3 operator*(const float3x3& m, const float3& v) {
    return make_float3(m.rows[0].x * v.x + m.rows[0].y * v.y + m.rows[0].z * v.z,
                       m.rows[1].x * v.x + m.rows[1].y * v.y + m.rows[1].z * v.z,
                       m.rows[2].x * v.x + m.rows[2].y * v.y + m.rows[2].z * v.z);
}

__host__ __device__ inline float3x3 inverse(const float3x3& m) {
    float3 a = m.rows[0];
    float3 b = m.rows[1];
    float3 c = m.rows[2];
    float3 r0 = cross(b, c);
    float3 r1 = cross(c, a);
    float3 r2 = cross(a, b);
    float det = dot(a, r0);
    if (fabsf(det) < 1e-8f)
        return float3x3(0.0f);
    float invDet = 1.0f / det;
    float3x3 inv;
    inv.rows[0] = r0 * invDet;
    inv.rows[1] = r1 * invDet;
    inv.rows[2] = r2 * invDet;
    return transpose(inv);
}

struct float4x4 {
    float4 rows[4];

    __host__ __device__ float4x4(float diag = 1.0f) {
        for (int r = 0; r < 4; ++r) {
            rows[r] = make_float4(0.0f);
            ((float*)&rows[r])[r] = diag;
        }
    }

    __host__ __device__ float4x4(const float4& r0, const float4& r1, const float4& r2, const float4& r3) {
        rows[0] = r0;
        rows[1] = r1;
        rows[2] = r2;
        rows[3] = r3;
    }

    __host__ __device__ float* operator[](int row) { return (float*)&rows[row]; }
    __host__ __device__ const float* operator[](int row) const { return (const float*)&rows[row]; }
};

__host__ __device__ inline float4x4 identity4x4() {
    return float4x4(1.0f);
}

__host__ __device__ inline float4x4 transpose(const float4x4& m) {
    float4x4 r(0.0f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ((float*)&r.rows[i])[j] = ((const float*)&m.rows[j])[i];
    return r;
}

__host__ __device__ inline float4x4 operator*(const float4x4& a, const float4x4& b) {
    float4x4 r(0.0f);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ((float*)&r.rows[i])[j] =
                    a.rows[i].x * ((const float*)&b.rows[0])[j] + a.rows[i].y * ((const float*)&b.rows[1])[j] +
                    a.rows[i].z * ((const float*)&b.rows[2])[j] + a.rows[i].w * ((const float*)&b.rows[3])[j];
        }
    }
    return r;
}

__host__ __device__ inline float4 operator*(const float4x4& m, const float4& v) {
    return make_float4(m.rows[0].x * v.x + m.rows[0].y * v.y + m.rows[0].z * v.z + m.rows[0].w * v.w,
                       m.rows[1].x * v.x + m.rows[1].y * v.y + m.rows[1].z * v.z + m.rows[1].w * v.w,
                       m.rows[2].x * v.x + m.rows[2].y * v.y + m.rows[2].z * v.z + m.rows[2].w * v.w,
                       m.rows[3].x * v.x + m.rows[3].y * v.y + m.rows[3].z * v.z + m.rows[3].w * v.w);
}

__host__ __device__ inline float3 transform_point(const float4x4& m, const float3& p) {
    float4 v = make_float4(p.x, p.y, p.z, 1.0f);
    float4 r = m * v;
    return make_float3(r.x / r.w, r.y / r.w, r.z / r.w);
}

__host__ __device__ inline float3 transform_vector(const float4x4& m, const float3& v) {
    float4 r = m * make_float4(v.x, v.y, v.z, 0.0f);
    return make_float3(r.x, r.y, r.z);
}

__host__ __device__ inline float4x4 inverse(const float4x4& m) {
    const float4& a = m.rows[0];
    const float4& b = m.rows[1];
    const float4& c = m.rows[2];
    const float4& d = m.rows[3];
    float3 a3 = make_float3(a.x, a.y, a.z);
    float3 b3 = make_float3(b.x, b.y, b.z);
    float3 c3 = make_float3(c.x, c.y, c.z);
    float3 d3 = make_float3(d.x, d.y, d.z);
    float3 s = cross(a3, b3);
    float3 t = cross(c3, d3);
    float3 u = a3 * b.w - b3 * a.w;
    float3 v = c3 * d.w - d3 * c.w;
    float det = dot(s, v) + dot(t, u);
    if (fabsf(det) < 1e-8f)
        return float4x4(0.0f);
    float invDet = 1.0f / det;
    s *= invDet;
    t *= invDet;
    u *= invDet;
    v *= invDet;
    float3 r0 = cross(b3, v) + t * b.w;
    float3 r1 = cross(v, a3) - t * a.w;
    float3 r2 = cross(d3, u) + s * d.w;
    float3 r3 = cross(u, c3) - s * c.w;
    float4x4 inv(0.0f);
    inv.rows[0] = make_float4(r0.x, r0.y, r0.z, -dot(b3, t));
    inv.rows[1] = make_float4(r1.x, r1.y, r1.z, dot(a3, t));
    inv.rows[2] = make_float4(r2.x, r2.y, r2.z, -dot(d3, s));
    inv.rows[3] = make_float4(r3.x, r3.y, r3.z, dot(c3, s));
    return transpose(inv);
}

#ifndef __DEVICE_BUILD__

__host__ inline float3x3 to_float3x3(const glm::mat3& m) {
    float3x3 r(0.0f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ((float*)&r.rows[i])[j] = m[j][i];
    return r;
}

__host__ inline float4x4 to_float4x4(const glm::mat4& m) {
    float4x4 r(0.0f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ((float*)&r.rows[i])[j] = m[j][i];
    return r;
}

#endif
