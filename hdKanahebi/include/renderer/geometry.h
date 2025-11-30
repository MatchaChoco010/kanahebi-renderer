#pragma once

#include <map>
#include <vector>

#include <optix.h>

#include <glm/glm.hpp>

#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include <oxu/cuda/buffer.h>
#include <oxu/optix/context.h>
#include <oxu/utils/error_check.h>

#include "renderer/common.h"
#include "renderer/shared.h"

using namespace pxr;
using namespace oxu::utils;

struct Geometry {
    Geometry(std::vector<glm::vec3>&& vertices,
             std::vector<glm::vec3>&& normals,
             std::vector<glm::vec2>&& uvs,
             std::vector<glm::vec3>&& colors,
             Interpolation normals_interpolation,
             Interpolation uvs_interpolation,
             Interpolation colors_interpolation,
             std::vector<uint32_t>&& indices,
             std::vector<uint32_t>&& face_indices,
             oxu::cuda::Buffer&& vertices_buffer,
             oxu::cuda::Buffer&& normals_buffer,
             oxu::cuda::Buffer&& uvs_buffer,
             oxu::cuda::Buffer&& colors_buffer,
             oxu::cuda::Buffer&& indices_buffer,
             oxu::cuda::Buffer&& face_indices_buffer,
             oxu::cuda::Buffer&& gas_buffer,
             OptixTraversableHandle gas_handle);

    static Geometry create(oxu::optix::Context& optix_context,
                           std::vector<glm::vec3>&& vertices,
                           std::vector<glm::vec3>&& normals,
                           std::vector<glm::vec2>&& uvs,
                           std::vector<glm::vec3>&& colors,
                           Interpolation normals_interpolation,
                           Interpolation uvs_interpolation,
                           Interpolation colors_interpolation,
                           std::vector<uint32_t>&& indices,
                           std::vector<uint32_t>&& face_indices);

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> colors;
    Interpolation normals_interpolation;
    Interpolation uvs_interpolation;
    Interpolation colors_interpolation;

    bool has_uvs() const { return !uvs.empty(); }
    bool has_colors() const { return !colors.empty(); }

    std::vector<uint32_t> indices;
    std::vector<uint32_t> face_indices;

    oxu::cuda::Buffer vertices_buffer;
    oxu::cuda::Buffer normals_buffer;
    oxu::cuda::Buffer uvs_buffer;
    oxu::cuda::Buffer colors_buffer;
    oxu::cuda::Buffer indices_buffer;
    oxu::cuda::Buffer face_indices_buffer;

    oxu::cuda::Buffer gas_buffer;
    OptixTraversableHandle gas_handle;
};

class GeometryManager {
public:
    GeometryManager(oxu::optix::Context& optix) : optix_(optix) {};

    void add_geometry(const SdfPath& prim_path,
                      std::vector<glm::vec3>&& vertices,
                      std::vector<glm::vec3>&& normals,
                      std::vector<glm::vec2>&& uvs,
                      std::vector<glm::vec3>&& colors,
                      Interpolation normals_interpolation,
                      Interpolation uvs_interpolation,
                      Interpolation colors_interpolation,
                      std::vector<uint32_t>&& indices,
                      std::vector<uint32_t>&& face_indices);

    void delete_geometry(const SdfPath& prim_path);

    const Geometry& get_geometry(const SdfPath& prim_path) const;

    BuildGeometryResult build_geometries() const;

private:
    oxu::optix::Context& optix_;
    std::map<SdfPath, Geometry> geometries_;
};
