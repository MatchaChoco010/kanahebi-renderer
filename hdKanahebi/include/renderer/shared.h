#pragma once

#include <cuda.h>
#include <math/matrix.h>
#include <optix.h>

struct SbtRecordHeader {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
template <typename T>
struct SbtRecord : SbtRecordHeader {
    T data;
};

struct EmptyData {};

struct HitGroupData {
    float4x4 transform;
    uint32_t geometry_index;
    uint32_t material_index;
};

struct CameraParams {
    float4x4 transform;
    float fov_y;
};

struct SceneParams {
    CUdeviceptr geometries;
    OptixTraversableHandle handle;
    CUdeviceptr materials;
    CameraParams camera_params;
};

struct Params {
    CUdeviceptr color_image;
    CUdeviceptr depth_image;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t N;
    uint32_t max_depth;
    uint32_t film_transparent;
    float exposure;
    CUdeviceptr scene_params;
    CUdeviceptr sobol_data;
};

enum class Interpolation : uint32_t {
    Constant = 0,
    Uniform,
    Vertex,
    FaceVarying,
    Instance,
};

struct GeometryData {
    CUdeviceptr vertices;
    CUdeviceptr normals;
    CUdeviceptr uvs;
    CUdeviceptr colors;
    Interpolation normals_interpolation;
    Interpolation uvs_interpolation;
    Interpolation colors_interpolation;
    CUdeviceptr indices;
    CUdeviceptr face_indices;
};

struct MaterialParams {
    float3 base_color;
    float metallic;
    float roughness;
    float3 emissive_color;
    float opacity;
    float ior;

    bool has_base_color_texture = false;
    bool has_normal_texture = false;
    bool has_metallic_texture = false;
    bool has_roughness_texture = false;
    bool has_emissive_color_texture = false;

    CUtexObject base_color_texture;
    CUtexObject normal_texture;
    CUtexObject metallic_texture;
    CUtexObject roughness_texture;
    CUtexObject emissive_color_texture;
};
