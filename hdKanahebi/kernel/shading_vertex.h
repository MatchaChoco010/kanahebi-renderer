#pragma once

#include "math/matrix.h"
#include "math/vector.h"

#include "params.h"
#include "trace.h"

struct ShadingVertex {
    float3 position;
    float3 geometry_normal;
    float3 shading_normal;
    float3 color;
    float2 uv;
    float3x3 tangent_frame;
};

__device__ ShadingVertex get_shading_vertex(const HitInfo& hit_info) {
    ShadingVertex vtx;

    SceneParams* scene_params = reinterpret_cast<SceneParams*>(params.scene_params);

    GeometryData* geometries = reinterpret_cast<GeometryData*>(scene_params->geometries);
    GeometryData& geometry_data = geometries[hit_info.geometry_index];

    uint32_t* indices = reinterpret_cast<uint32_t*>(geometry_data.indices);
    uint32_t* face_indices = reinterpret_cast<uint32_t*>(geometry_data.face_indices);

    float3* vertex_buffer = reinterpret_cast<float3*>(geometry_data.vertices);
    float3 v0 = vertex_buffer[indices[hit_info.primitive_index * 3 + 0]];
    float3 v1 = vertex_buffer[indices[hit_info.primitive_index * 3 + 1]];
    float3 v2 = vertex_buffer[indices[hit_info.primitive_index * 3 + 2]];

    float3* normal_buffer = reinterpret_cast<float3*>(geometry_data.normals);
    float3 n1;
    float3 n2;
    float3 n3;
    if (geometry_data.normals_interpolation == Interpolation::Vertex) {
        n1 = normal_buffer[indices[hit_info.primitive_index * 3 + 0]];
        n2 = normal_buffer[indices[hit_info.primitive_index * 3 + 1]];
        n3 = normal_buffer[indices[hit_info.primitive_index * 3 + 2]];
    } else if (geometry_data.normals_interpolation == Interpolation::FaceVarying) {
        n1 = normal_buffer[3 * hit_info.primitive_index + 0];
        n2 = normal_buffer[3 * hit_info.primitive_index + 1];
        n3 = normal_buffer[3 * hit_info.primitive_index + 2];
    }

    float2* uv_buffer = reinterpret_cast<float2*>(geometry_data.uvs);
    float2 uv1 = make_float2(0.0f, 0.0f);
    float2 uv2 = make_float2(0.0f, 0.0f);
    float2 uv3 = make_float2(0.0f, 0.0f);
    if (geometry_data.uvs_interpolation == Interpolation::Vertex) {
        uv1 = uv_buffer[indices[hit_info.primitive_index * 3 + 0]];
        uv2 = uv_buffer[indices[hit_info.primitive_index * 3 + 1]];
        uv3 = uv_buffer[indices[hit_info.primitive_index * 3 + 2]];
    } else if (geometry_data.uvs_interpolation == Interpolation::FaceVarying) {
        uv1 = uv_buffer[3 * hit_info.primitive_index + 0];
        uv2 = uv_buffer[3 * hit_info.primitive_index + 1];
        uv3 = uv_buffer[3 * hit_info.primitive_index + 2];
    }

    float3* color_buffer = reinterpret_cast<float3*>(geometry_data.colors);
    float3 c1;
    float3 c2;
    float3 c3;
    if (geometry_data.colors_interpolation == Interpolation::Constant) {
        c1 = color_buffer[0];
        c2 = color_buffer[0];
        c3 = color_buffer[0];
    } else if (geometry_data.colors_interpolation == Interpolation::Uniform) {
        c1 = color_buffer[face_indices[hit_info.primitive_index]];
        c2 = color_buffer[face_indices[hit_info.primitive_index]];
        c3 = color_buffer[face_indices[hit_info.primitive_index]];
    } else if (geometry_data.colors_interpolation == Interpolation::Vertex) {
        c1 = color_buffer[indices[hit_info.primitive_index * 3 + 0]];
        c2 = color_buffer[indices[hit_info.primitive_index * 3 + 1]];
        c3 = color_buffer[indices[hit_info.primitive_index * 3 + 2]];
    } else if (geometry_data.colors_interpolation == Interpolation::FaceVarying) {
        c1 = color_buffer[3 * hit_info.primitive_index + 0];
        c2 = color_buffer[3 * hit_info.primitive_index + 1];
        c3 = color_buffer[3 * hit_info.primitive_index + 2];
    }

    float3 position = hit_info.barycentric_coords.x * v0 + hit_info.barycentric_coords.y * v1 +
                      hit_info.barycentric_coords.z * v2;
    float3 geometry_normal = normalize(cross(v1 - v0, v2 - v0));
    float3 shading_normal = normalize(hit_info.barycentric_coords.x * n1 + hit_info.barycentric_coords.y * n2 +
                                      hit_info.barycentric_coords.z * n3);

    vtx.position = transform_point(hit_info.transform, position);

    float4x4 inv_transpose_world = inverse(transpose(hit_info.transform));
    vtx.geometry_normal = normalize(transform_vector(inv_transpose_world, geometry_normal));
    vtx.shading_normal = normalize(transform_vector(inv_transpose_world, shading_normal));

    vtx.uv = hit_info.barycentric_coords.x * uv1 + hit_info.barycentric_coords.y * uv2 +
             hit_info.barycentric_coords.z * uv3;

    vtx.uv.y = 1.0f - vtx.uv.y;  // V座標を反転

    vtx.color = hit_info.barycentric_coords.x * c1 + hit_info.barycentric_coords.y * c2 +
                hit_info.barycentric_coords.z * c3;

    float3 tangent;
    float3 bitangent;
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(dot(vtx.shading_normal, up)) > 0.999f) {
        up = make_float3(1.0f, 0.0f, 0.0f);
    }
    tangent = normalize(cross(up, vtx.shading_normal));
    bitangent = normalize(cross(vtx.shading_normal, tangent));
    vtx.tangent_frame = float3x3(tangent, bitangent, vtx.shading_normal);

    return vtx;
}
