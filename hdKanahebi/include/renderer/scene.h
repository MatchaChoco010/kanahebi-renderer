#pragma once

#include <optix.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/pxr.h>

#include <oxu/optix/context.h>

#include "renderer/camera.h"
#include "renderer/common.h"
#include "renderer/geometry.h"
#include "renderer/material.h"
#include "renderer/mesh.h"
#include "renderer/node_tree.h"
#include "renderer/renderer.h"
#include "renderer/shared.h"

using namespace pxr;

class Scene {
public:
    Scene(oxu::optix::Context& optix_context) : geometry_manager_(optix_context), optix_context_(optix_context) {};

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    Scene(const Scene&&) = delete;
    Scene&& operator=(const Scene&&) = delete;

    /// @brief メインカメラのパスを設定
    void set_main_camera_path(const SdfPath& path) { camera_manager_.set_main_camera_path(path); }

    /// @brief Primを追加
    void add_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    /// @brief Primを削除
    void delete_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    /// @brief Primを更新
    void update_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    /// @brief シェーダバインディングテーブルとParamsを構築
    std::unique_ptr<const SceneData> build(const ProgramGroups& program_groups) const;

private:
    /// @brief Primを追加
    void add_prim_child(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    /// @brief Primを削除
    void delete_prim_child(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    NodeTree node_tree_;
    CameraManager camera_manager_;
    GeometryManager geometry_manager_;
    MeshManager mesh_manager_;
    MaterialManager material_manager_;
    oxu::cuda::Buffer sobol_buffer_;
    oxu::optix::Context& optix_context_;
};
