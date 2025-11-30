#include "renderer/geometry.h"

Geometry::Geometry(std::vector<glm::vec3>&& vertices,
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
                   OptixTraversableHandle gas_handle)
    : vertices(std::move(vertices)),
      normals(std::move(normals)),
      uvs(std::move(uvs)),
      colors(std::move(colors)),
      normals_interpolation(normals_interpolation),
      uvs_interpolation(uvs_interpolation),
      colors_interpolation(colors_interpolation),
      indices(std::move(indices)),
      face_indices(std::move(face_indices)),
      vertices_buffer(std::move(vertices_buffer)),
      normals_buffer(std::move(normals_buffer)),
      uvs_buffer(std::move(uvs_buffer)),
      colors_buffer(std::move(colors_buffer)),
      indices_buffer(std::move(indices_buffer)),
      face_indices_buffer(std::move(face_indices_buffer)),
      gas_buffer(std::move(gas_buffer)),
      gas_handle(gas_handle) {}

Geometry Geometry::create(oxu::optix::Context& optix_context,
                          std::vector<glm::vec3>&& vertices,
                          std::vector<glm::vec3>&& normals,
                          std::vector<glm::vec2>&& uvs,
                          std::vector<glm::vec3>&& colors,
                          Interpolation normals_interpolation,
                          Interpolation uvs_interpolation,
                          Interpolation colors_interpolation,
                          std::vector<uint32_t>&& indices,
                          std::vector<uint32_t>&& face_indices) {
    if (uvs.empty()) {
        uvs_interpolation = Interpolation::Constant;
        uvs.emplace_back(0.0f, 0.0f);
    }

    if (colors.empty()) {
        colors_interpolation = Interpolation::Constant;
        colors.emplace_back(1.0f, 1.0f, 1.0f);
    }

    oxu::cuda::Buffer vertices_buffer(vertices.size() * sizeof(glm::vec3));
    oxu::cuda::Buffer normals_buffer(normals.size() * sizeof(glm::vec3));
    oxu::cuda::Buffer uvs_buffer(uvs.size() * sizeof(glm::vec2));
    oxu::cuda::Buffer colors_buffer(colors.size() * sizeof(glm::vec3));
    oxu::cuda::Buffer indices_buffer(indices.size() * sizeof(uint32_t));
    oxu::cuda::Buffer face_indices_buffer(face_indices.size() * sizeof(uint32_t));

    vertices_buffer.upload(std::span<const glm::vec3>(vertices.data(), vertices.size()));
    normals_buffer.upload(std::span<const glm::vec3>(normals.data(), normals.size()));
    uvs_buffer.upload(std::span<const glm::vec2>(uvs.data(), uvs.size()));
    colors_buffer.upload(std::span<const glm::vec3>(colors.data(), colors.size()));
    indices_buffer.upload(std::span<const uint32_t>(indices.data(), indices.size()));
    face_indices_buffer.upload(std::span<const uint32_t>(face_indices.data(), face_indices.size()));

    std::vector<CUdeviceptr> vertices_buffers = {vertices_buffer.device_ptr()};

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = vertices_buffers.data();
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    build_input.triangleArray.indexBuffer = indices_buffer.device_ptr();
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size() / 3);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);

    unsigned int triFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    build_input.triangleArray.flags = &triFlags;
    build_input.triangleArray.numSbtRecords = 1;

    build_input.triangleArray.sbtIndexOffsetBuffer = 0;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accel_opt = {};
    accel_opt.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_opt.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_sizes{};
    optix_check(optixAccelComputeMemoryUsage(optix_context, &accel_opt, &build_input, 1, &gas_sizes));

    oxu::cuda::Buffer temp_buffer(gas_sizes.tempSizeInBytes);
    oxu::cuda::Buffer gas_buffer(gas_sizes.outputSizeInBytes);

    OptixTraversableHandle gas_handle = 0;
    optix_check(optixAccelBuild(optix_context,
                                /*stream*/ 0, &accel_opt, &build_input, 1, temp_buffer.device_ptr(),
                                gas_sizes.tempSizeInBytes, gas_buffer.device_ptr(), gas_sizes.outputSizeInBytes,
                                &gas_handle,
                                /*emittedProperties*/ nullptr, 0));

    return Geometry(std::move(vertices), std::move(normals), std::move(uvs), std::move(colors), normals_interpolation,
                    uvs_interpolation, colors_interpolation, std::move(indices), std::move(face_indices),
                    std::move(vertices_buffer), std::move(normals_buffer), std::move(uvs_buffer),
                    std::move(colors_buffer), std::move(indices_buffer), std::move(face_indices_buffer),
                    std::move(gas_buffer), gas_handle);
}

void GeometryManager::add_geometry(const SdfPath& path,
                                   std::vector<glm::vec3>&& vertices,
                                   std::vector<glm::vec3>&& normals,
                                   std::vector<glm::vec2>&& uvs,
                                   std::vector<glm::vec3>&& colors,
                                   Interpolation normals_interpolation,
                                   Interpolation uvs_interpolation,
                                   Interpolation colors_interpolation,
                                   std::vector<uint32_t>&& indices,
                                   std::vector<uint32_t>&& face_indices) {
    geometries_.emplace(path, Geometry::create(optix_, std::move(vertices), std::move(normals), std::move(uvs),
                                               std::move(colors), normals_interpolation, uvs_interpolation,
                                               colors_interpolation, std::move(indices), std::move(face_indices)));
}

void GeometryManager::delete_geometry(const SdfPath& path) {
    geometries_.erase(path);
}

const Geometry& GeometryManager::get_geometry(const SdfPath& prim_path) const {
    return geometries_.at(prim_path);
}

BuildGeometryResult GeometryManager::build_geometries() const {
    BuildGeometryResult result;

    std::vector<GeometryData> geometry_data_list;

    uint32_t index = 0;
    for (const auto& [prim_path, geometry] : geometries_) {
        result.geometry_index_map[prim_path] = index++;
        result.gas_handle_list.push_back(geometry.gas_handle);

        GeometryData geometry_data{};
        geometry_data.vertices = geometry.vertices_buffer.device_ptr();
        geometry_data.normals = geometry.normals_buffer.device_ptr();
        geometry_data.uvs = geometry.uvs_buffer.device_ptr();
        geometry_data.colors = geometry.colors_buffer.device_ptr();
        geometry_data.normals_interpolation = geometry.normals_interpolation;
        geometry_data.uvs_interpolation = geometry.uvs_interpolation;
        geometry_data.colors_interpolation = geometry.colors_interpolation;
        geometry_data.indices = geometry.indices_buffer.device_ptr();
        geometry_data.face_indices = geometry.face_indices_buffer.device_ptr();
        geometry_data_list.push_back(geometry_data);
    }

    result.geometry_list_buffer = oxu::cuda::Buffer(geometry_data_list.size() * sizeof(GeometryData));
    result.geometry_list_buffer.upload(
            std::span<const GeometryData>(geometry_data_list.data(), geometry_data_list.size()));

    return result;
}
