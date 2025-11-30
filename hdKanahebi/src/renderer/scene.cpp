#include "renderer/scene.h"

void Scene::add_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    HdSceneIndexPrim prim = scene_index->GetPrim(path);
    if (!prim.dataSource) {
        return;
    }

    node_tree_.add_node(path, scene_index);

    if (prim.primType == HdPrimTypeTokens->camera) {
        camera_manager_.add_camera(path, scene_index);
    }

    if (prim.primType == HdPrimTypeTokens->mesh) {
        mesh_manager_.add_mesh(path, scene_index, geometry_manager_);
    }

    // GeomSubsetのみ単体で追加された場合は親メッシュを生成し直す
    if (prim.primType == HdPrimTypeTokens->geomSubset) {
        SdfPath parent_path = path.GetParentPath();
        mesh_manager_.delete_mesh(parent_path, geometry_manager_);
        mesh_manager_.add_mesh(parent_path, scene_index, geometry_manager_);
    }

    if (prim.primType == HdPrimTypeTokens->material) {
        material_manager_.add_material(path, scene_index);
    }

    SdfPathVector child_paths = scene_index->GetChildPrimPaths(path);
    for (const SdfPath& child_path : child_paths) {
        add_prim_child(child_path, scene_index);
    }
}

void Scene::add_prim_child(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    HdSceneIndexPrim prim = scene_index->GetPrim(path);
    if (!prim.dataSource) {
        return;
    }

    node_tree_.add_node(path, scene_index);

    if (prim.primType == HdPrimTypeTokens->camera) {
        camera_manager_.add_camera(path, scene_index);
    }

    if (prim.primType == HdPrimTypeTokens->mesh) {
        mesh_manager_.add_mesh(path, scene_index, geometry_manager_);
    }

    // GeomSubsetは親のmeshに所属しているためスキップ

    if (prim.primType == HdPrimTypeTokens->material) {
        material_manager_.add_material(path, scene_index);
    }

    SdfPathVector child_paths = scene_index->GetChildPrimPaths(path);
    for (const SdfPath& child_path : child_paths) {
        add_prim_child(child_path, scene_index);
    }
}

void Scene::delete_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    Node node = node_tree_.get_node(path);
    for (const SdfPath& child_path : node.children) {
        delete_prim_child(child_path, scene_index);
    }
    node_tree_.delete_node(path);
    camera_manager_.delete_camera(path);
    mesh_manager_.delete_mesh(path, geometry_manager_);

    // GeomSubsetのみ単体で削除された場合は親メッシュを生成し直す必要がある
    if (path.GetParentPath() != SdfPath::EmptyPath()) {
        SdfPath parent_path = path.GetParentPath();
        HdSceneIndexPrim parent_prim = scene_index->GetPrim(parent_path);
        if (parent_prim.primType == HdPrimTypeTokens->mesh) {
            mesh_manager_.delete_mesh(parent_path, geometry_manager_);
            mesh_manager_.add_mesh(parent_path, scene_index, geometry_manager_);
        }
    }

    material_manager_.delete_material(path);
}

void Scene::delete_prim_child(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    Node node = node_tree_.get_node(path);
    for (const SdfPath& child_path : node.children) {
        delete_prim_child(child_path, scene_index);
    }
    node_tree_.delete_node(path);
    camera_manager_.delete_camera(path);
    mesh_manager_.delete_mesh(path, geometry_manager_);

    // GeomSubsetは親のmeshに所属しているためスキップ

    material_manager_.delete_material(path);
}

void Scene::update_prim(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    HdSceneIndexPrim prim = scene_index->GetPrim(path);

    node_tree_.update_node(path, scene_index);

    if (prim.primType == HdPrimTypeTokens->camera) {
        camera_manager_.update_camera(path, scene_index);
    }

    if (prim.primType == HdPrimTypeTokens->mesh) {
        mesh_manager_.delete_mesh(path, geometry_manager_);
        mesh_manager_.add_mesh(path, scene_index, geometry_manager_);
    }

    // GeomSubsetのみ単体で更新された場合は親メッシュを生成し直す必要がある
    if (prim.primType == HdPrimTypeTokens->geomSubset) {
        SdfPath parent_path = path.GetParentPath();
        mesh_manager_.delete_mesh(parent_path, geometry_manager_);
        mesh_manager_.add_mesh(parent_path, scene_index, geometry_manager_);
    }

    if (prim.primType == HdPrimTypeTokens->material) {
        material_manager_.delete_material(path);
        material_manager_.add_material(path, scene_index);
    }
}

std::unique_ptr<const SceneData> Scene::build(const ProgramGroups& program_groups) const {
    std::vector<MeshData> meshes = mesh_manager_.get_meshes();
    BuildGeometryResult geometries = geometry_manager_.build_geometries();
    BuildMaterialResult materials = material_manager_.build_materials();

    std::vector<OptixInstance> instances;
    std::vector<uint32_t> geometry_indices;
    std::vector<uint32_t> material_indices;
    std::vector<glm::mat4> instance_transforms;

    // meshesからインスタンスを登録
    for (size_t i = 0; i < meshes.size(); ++i) {
        glm::mat4 transform = node_tree_.get_global_transform(meshes[i].geometry_path);
        float transform_array[12] = {
                transform[0][0], transform[1][0], transform[2][0], transform[3][0], transform[0][1], transform[1][1],
                transform[2][1], transform[3][1], transform[0][2], transform[1][2], transform[2][2], transform[3][2],
        };

        if (!meshes[i].only_subsets) {
            OptixInstance instance = {};
            std::memcpy(instance.transform, transform_array, sizeof(float) * 12);
            instance.instanceId = static_cast<unsigned int>(instances.size());
            instance.sbtOffset = instances.size() * RAY_TYPE_COUNT;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.visibilityMask = 255;

            const SdfPath geometry_path = meshes[i].geometry_path;
            const Geometry& geometry = geometry_manager_.get_geometry(geometry_path);
            instance.traversableHandle = geometry.gas_handle;

            instances.emplace_back(instance);
            geometry_indices.emplace_back(geometries.geometry_index_map.at(geometry_path));
            material_indices.emplace_back(materials.material_index_map.at(meshes[i].material_path));
            instance_transforms.emplace_back(transform);
        }

        for (const auto& subset : meshes[i].geom_subsets) {
            OptixInstance instance = {};
            std::memcpy(instance.transform, transform_array, sizeof(float) * 12);
            instance.instanceId = static_cast<unsigned int>(instances.size());
            instance.sbtOffset = instances.size() * RAY_TYPE_COUNT;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.visibilityMask = 255;

            const SdfPath geometry_path = subset.geometry_path;
            const Geometry& geometry = geometry_manager_.get_geometry(geometry_path);
            instance.traversableHandle = geometry.gas_handle;

            instances.emplace_back(instance);
            geometry_indices.emplace_back(geometries.geometry_index_map.at(geometry_path));
            material_indices.emplace_back(materials.material_index_map.at(subset.material_path));
            instance_transforms.emplace_back(transform);
        }
    }

    // インスタンスバッファを作成
    oxu::cuda::Buffer instances_buffer(instances.size() * sizeof(OptixInstance));
    instances_buffer.upload(std::span<const OptixInstance>(instances.data(), instances.size()));

    // IAS用のビルド入力を作成
    OptixBuildInput ias_input = {};
    ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_input.instanceArray.instances = instances_buffer.device_ptr();
    ias_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    // IASをビルド
    OptixAccelBuildOptions ias_options = {};
    ias_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_sizes{};
    optix_check(optixAccelComputeMemoryUsage(optix_context_, &ias_options, &ias_input, 1, &ias_sizes));

    oxu::cuda::Buffer ias_temp_buffer(ias_sizes.tempSizeInBytes);
    oxu::cuda::Buffer ias_buffer(ias_sizes.outputSizeInBytes);

    OptixTraversableHandle ias_handle = 0;
    optix_check(optixAccelBuild(optix_context_,
                                /*stream*/ 0, &ias_options, &ias_input, 1, ias_temp_buffer.device_ptr(),
                                ias_sizes.tempSizeInBytes, ias_buffer.device_ptr(), ias_sizes.outputSizeInBytes,
                                &ias_handle,
                                /*emittedProperties*/ nullptr, 0));

    // SBTを作成
    SbtRecord<EmptyData> rg_record{};
    optix_check(optixSbtRecordPackHeader(program_groups.raygen, &rg_record));

    oxu::cuda::Buffer rg_record_buffer(sizeof(rg_record));
    rg_record_buffer.upload(std::span<const SbtRecord<EmptyData>>(&rg_record, 1));

    std::vector<SbtRecord<EmptyData>> ms_records(RAY_TYPE_COUNT);
    for (size_t i = 0; i < RAY_TYPE_COUNT; ++i) {
        optix_check(optixSbtRecordPackHeader(*program_groups.miss[i], &ms_records[i]));
    }
    oxu::cuda::Buffer ms_record_buffer(sizeof(SbtRecord<EmptyData>) * ms_records.size());
    ms_record_buffer.upload(std::span<const SbtRecord<EmptyData>>(ms_records.data(), ms_records.size()));

    std::vector<SbtRecord<HitGroupData>> hg_records(instances.size() * RAY_TYPE_COUNT);
    for (size_t i = 0; i < instances.size(); ++i) {
        for (size_t j = 0; j < RAY_TYPE_COUNT; ++j) {
            optix_check(optixSbtRecordPackHeader(*program_groups.hit[j], &hg_records[i * RAY_TYPE_COUNT + j]));
            HitGroupData hg_data;
            hg_data.transform = to_float4x4(instance_transforms[i]);
            hg_data.geometry_index = geometry_indices[i];
            hg_data.material_index = material_indices[i];
            hg_records[i * RAY_TYPE_COUNT + j].data = hg_data;
        }
    }
    oxu::cuda::Buffer hg_record_buffer(sizeof(SbtRecord<HitGroupData>) * hg_records.size());
    hg_record_buffer.upload(std::span<const SbtRecord<HitGroupData>>(hg_records.data(), hg_records.size()));

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = rg_record_buffer.device_ptr();
    sbt.missRecordBase = ms_record_buffer.device_ptr();
    sbt.missRecordStrideInBytes = sizeof(SbtRecord<EmptyData>);
    sbt.missRecordCount = static_cast<unsigned int>(RAY_TYPE_COUNT);
    sbt.hitgroupRecordBase = hg_record_buffer.device_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(instances.size() * RAY_TYPE_COUNT);

    // Paramsバッファを作成
    oxu::cuda::Buffer scene_params_buffer(sizeof(SceneParams));
    SceneParams scene_params;
    scene_params.handle = ias_handle;
    scene_params.geometries = geometries.geometry_list_buffer.device_ptr();
    scene_params.materials = materials.material_list_buffer.device_ptr();

    CameraData camera = camera_manager_.get_main_camera();
    scene_params.camera_params.transform = to_float4x4(camera.transform);
    scene_params.camera_params.fov_y = camera.fov_y;

    scene_params_buffer.upload(std::span<const SceneParams>(&scene_params, 1));

    return std::make_unique<SceneData>(std::move(scene_params_buffer), std::move(sbt), ias_handle,
                                       std::move(instances_buffer), std::move(ias_buffer), std::move(rg_record_buffer),
                                       std::move(ms_record_buffer), std::move(hg_record_buffer), std::move(geometries),
                                       std::move(materials));
}
