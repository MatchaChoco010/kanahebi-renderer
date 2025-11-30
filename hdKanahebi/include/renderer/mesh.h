#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/geomSubsetSchema.h>
#include <pxr/imaging/hd/materialBindingSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/meshTopologySchema.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>

#include "renderer/geometry.h"

using namespace pxr;

struct GeomSubsetData {
    SdfPath geometry_path;
    SdfPath material_path;
};

struct MeshData {
    SdfPath geometry_path;
    bool only_subsets = false;
    SdfPath material_path;
    std::vector<GeomSubsetData> geom_subsets;
};

class MeshManager {
public:
    void add_mesh(const SdfPath& path, const HdSceneIndexBasePtr& scene_index, GeometryManager& geometry_manager);

    void delete_mesh(const SdfPath& path, GeometryManager& geometry_manager);

    std::vector<MeshData> get_meshes() const;

private:
    std::map<SdfPath, MeshData> meshes_;

    // ヘルパー
    struct SubsetTopologyData {
        HdMeshTopology topology;
        VtIntArray vertex_map;  // new_vertex_idx -> original_vertex_idx
    };

    /// @brief スムース法線を計算する
    static std::vector<glm::vec3> compute_smooth_normals(const std::vector<glm::vec3>& vertices,
                                                         const std::vector<uint32_t>& triangulated_indices);

    /// @brief TfTokenをInterpolationに変換
    static Interpolation token_to_interpolation(const pxr::TfToken& token);

    /// @brief GeomSubsetのパスを取得
    static std::vector<pxr::SdfPath> get_geom_subset_paths(const pxr::HdSceneIndexBaseRefPtr& scene_index,
                                                           const pxr::SdfPath& prim_path);

    /// @brief サブセット用のトポロジーを作成
    static SubsetTopologyData create_subset_topology(const HdMeshTopology& original_topology,
                                                     const VtIntArray& subset_face_indices);
};
