#pragma once

#include <array>

#include <optix.h>

#include <oxu/cuda/buffer.h>
#include <oxu/optix/context.h>
#include <oxu/optix/program_group.h>

using namespace pxr;

constexpr int RAY_TYPE_COUNT = 1;

struct BuildGeometryResult {
    std::map<SdfPath, uint32_t> geometry_index_map;
    std::vector<OptixTraversableHandle> gas_handle_list;
    oxu::cuda::Buffer geometry_list_buffer;
};

struct BuildMaterialResult {
    std::map<SdfPath, uint32_t> material_index_map;
    oxu::cuda::Buffer material_list_buffer;
};

struct SceneData {
    SceneData(oxu::cuda::Buffer&& scene_params_buffer,
              OptixShaderBindingTable&& sbt,
              OptixTraversableHandle ias_handle,
              oxu::cuda::Buffer&& instances_buffer,
              oxu::cuda::Buffer&& ias_buffer,
              oxu::cuda::Buffer&& raygen_record_buffer,
              oxu::cuda::Buffer&& miss_record_buffer,
              oxu::cuda::Buffer&& hitgroup_record_buffer,
              BuildGeometryResult&& geometries,
              BuildMaterialResult&& materials)
        : scene_params_buffer(std::move(scene_params_buffer)),
          sbt(std::move(sbt)),
          ias_handle_(ias_handle),
          instances_buffer_(std::move(instances_buffer)),
          ias_buffer_(std::move(ias_buffer)),
          raygen_record_buffer_(std::move(raygen_record_buffer)),
          miss_record_buffer_(std::move(miss_record_buffer)),
          hitgroup_record_buffer_(std::move(hitgroup_record_buffer)),
          geometries_(std::move(geometries)),
          materials_(std::move(materials)) {}

    oxu::cuda::Buffer scene_params_buffer;
    OptixShaderBindingTable sbt;

private:
    OptixTraversableHandle ias_handle_;
    oxu::cuda::Buffer instances_buffer_;
    oxu::cuda::Buffer ias_buffer_;
    oxu::cuda::Buffer raygen_record_buffer_;
    oxu::cuda::Buffer miss_record_buffer_;
    oxu::cuda::Buffer hitgroup_record_buffer_;
    BuildGeometryResult geometries_;
    BuildMaterialResult materials_;
};

struct ProgramGroups {
    const oxu::optix::ProgramGroup& raygen;
    const std::array<const oxu::optix::ProgramGroup*, RAY_TYPE_COUNT> miss;
    const std::array<const oxu::optix::ProgramGroup*, RAY_TYPE_COUNT> hit;
};
