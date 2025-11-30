#pragma once
#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ inline float2 make_float2(float s) {
    return make_float2(s, s);
}
__host__ __device__ inline float3 make_float3(float s) {
    return make_float3(s, s, s);
}
__host__ __device__ inline float4 make_float4(float s) {
    return make_float4(s, s, s, s);
}

//-----------------------------
// float2 operators
//-----------------------------
__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline float2 operator-(const float2& a) {
    return make_float2(-a.x, -a.y);
}
__host__ __device__ inline float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline float2 operator/(const float2& a, const float2& b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline float2 operator+(const float2& a, float s) {
    return make_float2(a.x + s, a.y + s);
}
__host__ __device__ inline float2 operator+(float s, const float2& a) {
    return a + s;
}
__host__ __device__ inline float2 operator-(const float2& a, float s) {
    return make_float2(a.x - s, a.y - s);
}
__host__ __device__ inline float2 operator-(float s, const float2& a) {
    return make_float2(s - a.x, s - a.y);
}
__host__ __device__ inline float2 operator*(const float2& a, float s) {
    return make_float2(a.x * s, a.y * s);
}
__host__ __device__ inline float2 operator*(float s, const float2& a) {
    return a * s;
}
__host__ __device__ inline float2 operator/(const float2& a, float s) {
    float inv = 1.0f / s;
    return make_float2(a.x * inv, a.y * inv);
}
__host__ __device__ inline float2 operator/(float s, const float2& a) {
    return make_float2(s / a.x, s / a.y);
}

__host__ __device__ inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}
__host__ __device__ inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
__host__ __device__ inline float2& operator*=(float2& a, const float2& b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}
__host__ __device__ inline float2& operator/=(float2& a, const float2& b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
}

__host__ __device__ inline float2& operator+=(float2& a, float s) {
    a.x += s;
    a.y += s;
    return a;
}
__host__ __device__ inline float2& operator-=(float2& a, float s) {
    a.x -= s;
    a.y -= s;
    return a;
}
__host__ __device__ inline float2& operator*=(float2& a, float s) {
    a.x *= s;
    a.y *= s;
    return a;
}
__host__ __device__ inline float2& operator/=(float2& a, float s) {
    float inv = 1.0f / s;
    a.x *= inv;
    a.y *= inv;
    return a;
}

__host__ __device__ inline float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}
__host__ __device__ inline float length(const float2& v) {
    return sqrtf(dot(v, v));
}
__host__ __device__ inline float2 normalize(const float2& v) {
    float l = length(v);
    return (l > 1e-8f) ? v / l : make_float2(0.0f);
}
__host__ __device__ inline float2 powf(const float2& v, const float& e) {
    return make_float2(powf(v.x, e), powf(v.y, e));
}

//-----------------------------
// float3 operators
//-----------------------------
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline float3 operator+(const float3& a, float s) {
    return make_float3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline float3 operator+(float s, const float3& a) {
    return a + s;
}
__host__ __device__ inline float3 operator-(const float3& a, float s) {
    return make_float3(a.x - s, a.y - s, a.z - s);
}
__host__ __device__ inline float3 operator-(float s, const float3& a) {
    return make_float3(s - a.x, s - a.y, s - a.z);
}
__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return a * s;
}
__host__ __device__ inline float3 operator/(const float3& a, float s) {
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}
__host__ __device__ inline float3 operator/(float s, const float3& a) {
    return make_float3(s / a.x, s / a.y, s / a.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
__host__ __device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}
__host__ __device__ inline float3& operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

__host__ __device__ inline float3& operator+=(float3& a, float s) {
    a.x += s;
    a.y += s;
    a.z += s;
    return a;
}
__host__ __device__ inline float3& operator-=(float3& a, float s) {
    a.x -= s;
    a.y -= s;
    a.z -= s;
    return a;
}
__host__ __device__ inline float3& operator*=(float3& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}
__host__ __device__ inline float3& operator/=(float3& a, float s) {
    float inv = 1.0f / s;
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    return a;
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}
__host__ __device__ inline float3 normalize(const float3& v) {
    float l = length(v);
    return (l > 1e-8f) ? v / l : make_float3(0.0f);
}
__host__ __device__ inline float3 powf(const float3& v, const float& e) {
    return make_float3(powf(v.x, e), powf(v.y, e), powf(v.z, e));
}
__host__ __device__ inline float3 reflect(const float3& I, const float3& N) {
    return I - 2.0f * dot(N, I) * N;
}

//-----------------------------
// float4 operators
//-----------------------------
__host__ __device__ inline float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline float4 operator-(const float4& a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
__host__ __device__ inline float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ inline float4 operator+(const float4& a, float s) {
    return make_float4(a.x + s, a.y + s, a.z + s, a.w + s);
}
__host__ __device__ inline float4 operator+(float s, const float4& a) {
    return a + s;
}
__host__ __device__ inline float4 operator-(const float4& a, float s) {
    return make_float4(a.x - s, a.y - s, a.z - s, a.w - s);
}
__host__ __device__ inline float4 operator-(float s, const float4& a) {
    return make_float4(s - a.x, s - a.y, s - a.z, s - a.w);
}
__host__ __device__ inline float4 operator*(const float4& a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__host__ __device__ inline float4 operator*(float s, const float4& a) {
    return a * s;
}
__host__ __device__ inline float4 operator/(const float4& a, float s) {
    float inv = 1.0f / s;
    return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}
__host__ __device__ inline float4 operator/(float s, const float4& a) {
    return make_float4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__host__ __device__ inline float4& operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
__host__ __device__ inline float4& operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
__host__ __device__ inline float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}
__host__ __device__ inline float4& operator/=(float4& a, const float4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    return a;
}

__host__ __device__ inline float4& operator+=(float4& a, float s) {
    a.x += s;
    a.y += s;
    a.z += s;
    a.w += s;
    return a;
}
__host__ __device__ inline float4& operator-=(float4& a, float s) {
    a.x -= s;
    a.y -= s;
    a.z -= s;
    a.w -= s;
    return a;
}
__host__ __device__ inline float4& operator*=(float4& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
    return a;
}
__host__ __device__ inline float4& operator/=(float4& a, float s) {
    float inv = 1.0f / s;
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    a.w *= inv;
    return a;
}

__host__ __device__ inline float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__host__ __device__ inline float length(const float4& v) {
    return sqrtf(dot(v, v));
}
__host__ __device__ inline float4 normalize(const float4& v) {
    float l = length(v);
    return (l > 1e-8f) ? v / l : make_float4(0.0f);
}
__host__ __device__ inline float4 powf(const float4& v, const float& e) {
    return make_float4(powf(v.x, e), powf(v.y, e), powf(v.z, e), powf(v.w, e));
}
