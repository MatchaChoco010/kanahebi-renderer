#include "renderer/mesh.h"

void MeshManager::add_mesh(const SdfPath& prim_path,
                           const HdSceneIndexBasePtr& scene_index,
                           GeometryManager& geometry_manager) {
    // メッシュを作成
    MeshData new_mesh;
    new_mesh.geometry_path = prim_path;

    // Scene Indexからプリムデータを取得
    HdSceneIndexPrim prim = scene_index->GetPrim(prim_path);
    if (!prim.dataSource) {
        throw std::runtime_error("Invalid prim at prim_path: " + prim_path.GetString());
    }

    // メッシュプリムであることを確認
    if (prim.primType != HdPrimTypeTokens->mesh) {
        throw std::runtime_error("Prim at prim_path is not a mesh: " + prim_path.GetString());
    }

    HdContainerDataSourceHandle prim_ds = prim.dataSource;

    HdDataSourceBaseHandle mesh_ds_base = prim_ds->Get(TfToken("mesh"));
    if (!mesh_ds_base) {
        throw std::runtime_error("No mesh data source found at prim_path: " + prim_path.GetString());
    }

    HdContainerDataSourceHandle mesh_ds = HdContainerDataSource::Cast(mesh_ds_base);
    if (!mesh_ds) {
        throw std::runtime_error("Mesh data source is not a container at prim_path: " + prim_path.GetString());
    }

    // Topology情報の取得
    HdMeshTopologySchema topology_schema = HdMeshTopologySchema::GetFromParent(mesh_ds);
    if (!topology_schema) {
        throw std::runtime_error("No mesh topology found at prim_path: " + prim_path.GetString());
    }

    // Face vertex countsを取得
    HdIntArrayDataSourceHandle fvc_ds = topology_schema.GetFaceVertexCounts();
    VtIntArray face_vertex_counts = fvc_ds->GetTypedValue(0.0f);

    // 面が存在しない場合はMeshを登録せず終了
    if (face_vertex_counts.size() == 0) {
        return;
    }

    // Face vertex indicesを取得
    HdIntArrayDataSourceHandle fvi_ds = topology_schema.GetFaceVertexIndices();
    VtIntArray face_vertex_indices = fvi_ds->GetTypedValue(0.0f);

    // Orientationを取得
    TfToken orientation = HdTokens->rightHanded;
    HdTokenDataSourceHandle orient_ds = topology_schema.GetOrientation();
    if (orient_ds) {
        orientation = orient_ds->GetTypedValue(0.0f);
    }

    // SchemeはデフォルトでCatmull-Clark
    TfToken scheme = PxOsdOpenSubdivTokens->catmullClark;

    // HdMeshTopologyを構築
    HdMeshTopology topology(scheme, orientation, face_vertex_counts, face_vertex_indices);

    // Pointsを取得
    HdPrimvarsSchema primvars_schema = HdPrimvarsSchema::GetFromParent(prim_ds);
    if (!primvars_schema) {
        throw std::runtime_error("No primvars schema found at prim_path: " + prim_path.GetString());
    }

    HdPrimvarSchema points_schema = primvars_schema.GetPrimvar(HdTokens->points);
    if (!points_schema) {
        throw std::runtime_error("No points primvar found");
    }

    HdSampledDataSourceHandle points_value_ds = points_schema.GetPrimvarValue();
    VtValue points_value = points_value_ds->GetValue(0.0f);
    VtVec3fArray points = points_value.Get<VtVec3fArray>();

    // その他のPrimvarの取得
    HdPrimvarSchema normals_schema = primvars_schema.GetPrimvar(HdTokens->normals);
    HdPrimvarSchema uvs_schema = primvars_schema.GetPrimvar(TfToken("st"));
    HdPrimvarSchema display_color_schema = primvars_schema.GetPrimvar(HdTokens->displayColor);

    // Primvarのinterpolationを取得
    Interpolation normals_interp = Interpolation::Vertex;
    Interpolation uvs_interp = Interpolation::FaceVarying;
    Interpolation colors_interp = Interpolation::Uniform;

    VtVec3fArray original_normals;
    VtVec2fArray original_uvs;
    VtVec3fArray original_colors;

    bool has_normals = false;
    bool has_uvs = false;
    bool has_colors = false;

    if (normals_schema) {
        HdSampledDataSourceHandle value_ds = normals_schema.GetPrimvarValue();
        VtValue value = value_ds->GetValue(0.0f);
        original_normals = value.Get<VtVec3fArray>();
        has_normals = true;

        HdTokenDataSourceHandle interp_ds = normals_schema.GetInterpolation();
        if (interp_ds) {
            normals_interp = token_to_interpolation(interp_ds->GetTypedValue(0.0f));
        }
    }

    if (uvs_schema) {
        HdSampledDataSourceHandle value_ds = uvs_schema.GetPrimvarValue();
        VtValue value = value_ds->GetValue(0.0f);
        original_uvs = value.Get<VtVec2fArray>();
        has_uvs = true;

        HdTokenDataSourceHandle interp_ds = uvs_schema.GetInterpolation();
        if (interp_ds) {
            uvs_interp = token_to_interpolation(interp_ds->GetTypedValue(0.0f));
        }
    }

    if (display_color_schema) {
        HdSampledDataSourceHandle value_ds = display_color_schema.GetPrimvarValue();
        VtValue value = value_ds->GetValue(0.0f);
        original_colors = value.Get<VtVec3fArray>();
        has_colors = true;

        HdTokenDataSourceHandle interp_ds = display_color_schema.GetInterpolation();
        if (interp_ds) {
            colors_interp = token_to_interpolation(interp_ds->GetTypedValue(0.0f));
        }
    }

    const VtIntArray& topology_face_vertex_counts = topology.GetFaceVertexCounts();
    const int num_faces = topology_face_vertex_counts.size();

    // 子primを辿ってGeomSubsetを探す
    std::vector<SdfPath> subset_paths = get_geom_subset_paths(scene_index, prim_path);

    std::unordered_set<int> used_face_indices;
    std::map<SdfPath, VtIntArray> subset_face_indices_map;

    for (size_t i = 0; i < subset_paths.size(); i++) {
        const SdfPath& subset_path = subset_paths[i];
        HdSceneIndexPrim subset_prim = scene_index->GetPrim(subset_path);

        if (!subset_prim.dataSource) {
            continue;
        }

        // GeomSubsetSchemaを取得
        HdGeomSubsetSchema subset_schema = HdGeomSubsetSchema::GetFromParent(subset_prim.dataSource);

        if (!subset_schema) {
            continue;
        }

        // 面インデックスを取得
        HdIntArrayDataSourceHandle indices_ds = subset_schema.GetIndices();
        if (!indices_ds) {
            continue;
        }

        VtIntArray face_indices = indices_ds->GetTypedValue(0.0f);

        // 使用された面を記録
        for (int face_idx : face_indices) {
            used_face_indices.insert(face_idx);
        }

        subset_face_indices_map[subset_path] = face_indices;

        // GeomSubsetDataを作成
        GeomSubsetData subset;
        subset.geometry_path = subset_path;
        new_mesh.geom_subsets.push_back(subset);
    }

    // 残りの面を見つける
    VtIntArray remaining_faces;
    for (int face_idx = 0; face_idx < num_faces; face_idx++) {
        if (used_face_indices.find(face_idx) == used_face_indices.end()) {
            remaining_faces.push_back(face_idx);
        }
    }

    if (!remaining_faces.empty()) {
        subset_face_indices_map[prim_path] = remaining_faces;
    }

    // サブセットが1つもない場合
    if (subset_face_indices_map.empty()) {
        VtIntArray all_faces;
        all_faces.reserve(num_faces);
        for (int i = 0; i < num_faces; i++) {
            all_faces.push_back(i);
        }
        subset_face_indices_map[prim_path] = all_faces;
    }

    // subset_face_indices_mapにprim_pathが含まれていない場合、
    // このmeshにはgeom_subsetしか存在しない
    if (subset_face_indices_map.find(prim_path) == subset_face_indices_map.end()) {
        new_mesh.only_subsets = true;
    }

    // 各サブセットを処理
    for (const auto& [subset_path, subset_face_indices] : subset_face_indices_map) {
        // サブセット用のトポロジーを作成
        SubsetTopologyData subset_topo_data = create_subset_topology(topology, subset_face_indices);
        HdMeshTopology subset_topology = subset_topo_data.topology;

        // 頂点マップを取得
        VtIntArray subset_vertex_map = subset_topo_data.vertex_map;

        // 頂点データを準備
        std::vector<glm::vec3> subset_vertices;
        subset_vertices.reserve(subset_vertex_map.size());

        for (int original_vertex_idx : subset_vertex_map) {
            GfVec3f original_vertex = points[original_vertex_idx];
            subset_vertices.push_back(glm::vec3(original_vertex[0], original_vertex[1], original_vertex[2]));
        }
        // 三角形化
        HdMeshUtil subset_mesh_util(&subset_topology, SdfPath("/temp"));

        VtVec3iArray original_triangulated_indices;
        VtIntArray primitive_params;
        subset_mesh_util.ComputeTriangleIndices(&original_triangulated_indices, &primitive_params, nullptr);

        std::vector<uint32_t> triangulated_indices;
        triangulated_indices.reserve(original_triangulated_indices.size());
        for (size_t i = 0; i < original_triangulated_indices.size(); i++) {
            GfVec3i tri = original_triangulated_indices[i];
            triangulated_indices.push_back(static_cast<uint32_t>(tri[0]));
            triangulated_indices.push_back(static_cast<uint32_t>(tri[1]));
            triangulated_indices.push_back(static_cast<uint32_t>(tri[2]));
        }

        std::vector<uint32_t> face_indices;
        face_indices.resize(primitive_params.size());
        for (size_t i = 0; i < primitive_params.size(); i++) {
            face_indices[i] = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(primitive_params[i]);
        }

        // Normalsを処理
        std::vector<glm::vec3> subset_normals;
        Interpolation subset_normals_interp;

        if (has_normals) {
            if (normals_interp == Interpolation::Vertex) {
                // Vertex interpolation: 頂点マップを使って抽出
                subset_normals.reserve(subset_vertex_map.size());
                for (int original_vertex_idx : subset_vertex_map) {
                    GfVec3f original_normal = original_normals[original_vertex_idx];
                    subset_normals.push_back(glm::vec3(original_normal[0], original_normal[1], original_normal[2]));
                }
                subset_normals_interp = Interpolation::Vertex;
            } else if (normals_interp == Interpolation::FaceVarying) {
                // FaceVarying: サブセットのUVと同じ処理
                VtVec3fArray subset_original_normals;

                for (int face_idx : subset_face_indices) {
                    int fv_count = topology_face_vertex_counts[face_idx];

                    int original_offset = 0;
                    for (int f = 0; f < face_idx; f++) {
                        original_offset += topology_face_vertex_counts[f];
                    }

                    for (int i = 0; i < fv_count; i++) {
                        subset_original_normals.push_back(original_normals[original_offset + i]);
                    }
                }

                // 三角形化
                VtValue triangulated_normals;
                subset_mesh_util.ComputeTriangulatedFaceVaryingPrimvar(subset_original_normals.data(),
                                                                       subset_original_normals.size(), HdTypeFloatVec3,
                                                                       &triangulated_normals);
                VtVec3fArray original_subset_normals = triangulated_normals.Get<VtVec3fArray>();
                subset_normals.reserve(original_subset_normals.size());
                for (const auto& n : original_subset_normals) {
                    subset_normals.push_back(glm::vec3(n[0], n[1], n[2]));
                }
                subset_normals_interp = Interpolation::FaceVarying;
            }
        } else {
            // Normalを計算
            subset_normals = compute_smooth_normals(subset_vertices, triangulated_indices);
            subset_normals_interp = Interpolation::Vertex;
        }

        // UVsを処理
        std::vector<glm::vec2> subset_uvs;
        Interpolation subset_uvs_interp;

        if (has_uvs) {
            if (uvs_interp == Interpolation::FaceVarying) {
                VtVec2fArray subset_original_uvs;

                for (int face_idx : subset_face_indices) {
                    int fv_count = topology_face_vertex_counts[face_idx];

                    int original_uv_offset = 0;
                    for (int f = 0; f < face_idx; f++) {
                        original_uv_offset += topology_face_vertex_counts[f];
                    }

                    for (int i = 0; i < fv_count; i++) {
                        subset_original_uvs.push_back(original_uvs[original_uv_offset + i]);
                    }
                }

                VtValue triangulated_uvs;
                subset_mesh_util.ComputeTriangulatedFaceVaryingPrimvar(
                        subset_original_uvs.data(), subset_original_uvs.size(), HdTypeFloatVec2, &triangulated_uvs);
                VtVec2fArray original_subset_uvs = triangulated_uvs.Get<VtVec2fArray>();
                subset_uvs.reserve(original_subset_uvs.size());
                for (const auto& uv : original_subset_uvs) {
                    subset_uvs.push_back(glm::vec2(uv[0], uv[1]));
                }
                subset_uvs_interp = Interpolation::FaceVarying;
            } else if (uvs_interp == Interpolation::Vertex) {
                subset_uvs.reserve(subset_vertex_map.size());
                for (int original_vertex_idx : subset_vertex_map) {
                    GfVec2f original_uv = original_uvs[original_vertex_idx];
                    subset_uvs.push_back(glm::vec2(original_uv[0], original_uv[1]));
                }
                subset_uvs_interp = Interpolation::Vertex;
            }
        }

        // Colorsを処理
        std::vector<glm::vec3> subset_colors;
        Interpolation subset_colors_interp;

        if (has_colors) {
            if (colors_interp == Interpolation::Uniform) {
                subset_colors.reserve(subset_face_indices.size());
                for (int face_idx : subset_face_indices) {
                    GfVec3f original_color = original_colors[face_idx];
                    subset_colors.push_back(glm::vec3(original_color[0], original_color[1], original_color[2]));
                }
                subset_colors_interp = Interpolation::Uniform;
            } else if (colors_interp == Interpolation::Vertex) {
                subset_colors.reserve(subset_vertex_map.size());
                for (int original_vertex_idx : subset_vertex_map) {
                    GfVec3f original_color = original_colors[original_vertex_idx];
                    subset_colors.push_back(glm::vec3(original_color[0], original_color[1], original_color[2]));
                }
                subset_colors_interp = Interpolation::Vertex;
            } else if (colors_interp == Interpolation::FaceVarying) {
                VtVec3fArray subset_original_colors;

                for (int face_idx : subset_face_indices) {
                    int fv_count = topology_face_vertex_counts[face_idx];

                    int original_offset = 0;
                    for (int f = 0; f < face_idx; f++) {
                        original_offset += topology_face_vertex_counts[f];
                    }

                    for (int i = 0; i < fv_count; i++) {
                        subset_original_colors.push_back(original_colors[original_offset + i]);
                    }
                }

                VtValue triangulated_colors;
                subset_mesh_util.ComputeTriangulatedFaceVaryingPrimvar(subset_original_colors.data(),
                                                                       subset_original_colors.size(), HdTypeFloatVec3,
                                                                       &triangulated_colors);
                VtVec3fArray original_subset_colors = triangulated_colors.Get<VtVec3fArray>();
                subset_colors.reserve(original_subset_colors.size());
                for (const auto& c : original_subset_colors) {
                    subset_colors.push_back(glm::vec3(c[0], c[1], c[2]));
                }
                subset_colors_interp = Interpolation::FaceVarying;
            }
        }

        // GeomSubsetをジオメトリマネージャに登録
        geometry_manager.add_geometry(subset_path, std::move(subset_vertices), std::move(subset_normals),
                                      std::move(subset_uvs), std::move(subset_colors), subset_normals_interp,
                                      subset_uvs_interp, subset_colors_interp, std::move(triangulated_indices),
                                      std::move(face_indices));
    }

    // Material Pathの取得
    HdMaterialBindingsSchema material_bindings_schema = HdMaterialBindingsSchema::GetFromParent(prim_ds);
    if (material_bindings_schema) {
        HdMaterialBindingSchema material_binding_schema = material_bindings_schema.GetMaterialBinding(TfToken("0"));
        SdfPath material_path = material_binding_schema.GetPath()->GetTypedValue(0.0f);
        new_mesh.material_path = material_path;
    } else {
        new_mesh.material_path = SdfPath::EmptyPath();
    }
    for (auto& subset : new_mesh.geom_subsets) {
        HdDataSourceBaseHandle subset_prim_ds_base = scene_index->GetPrim(subset.geometry_path).dataSource;
        if (!subset_prim_ds_base) {
            continue;
        }

        HdContainerDataSourceHandle subset_prim_ds = HdContainerDataSource::Cast(subset_prim_ds_base);
        if (!subset_prim_ds) {
            continue;
        }

        HdMaterialBindingsSchema subset_material_bindings_schema =
                HdMaterialBindingsSchema::GetFromParent(subset_prim_ds);
        if (subset_material_bindings_schema) {
            HdMaterialBindingSchema subset_material_binding_schema =
                    subset_material_bindings_schema.GetMaterialBinding(TfToken("0"));
            SdfPath subset_material_path = subset_material_binding_schema.GetPath()->GetTypedValue(0.0f);
            subset.material_path = subset_material_path;
        } else {
            subset.material_path = SdfPath::EmptyPath();
        }
    }

    // メッシュを登録
    meshes_.emplace(prim_path, std::move(new_mesh));
}

void MeshManager::delete_mesh(const SdfPath& prim_path, GeometryManager& geometry_manager) {
    auto it = meshes_.find(prim_path);
    if (it != meshes_.end()) {
        // 関連ジオメトリの削除
        geometry_manager.delete_geometry(it->second.geometry_path);
        for (const auto& subset : it->second.geom_subsets) {
            geometry_manager.delete_geometry(subset.geometry_path);
        }

        // メッシュの削除
        meshes_.erase(it);
    }
}

std::vector<MeshData> MeshManager::get_meshes() const {
    std::vector<MeshData> mesh_list;
    for (const auto& [path, mesh] : meshes_) {
        mesh_list.push_back(mesh);
    }
    return mesh_list;
}

std::vector<glm::vec3> MeshManager::compute_smooth_normals(const std::vector<glm::vec3>& vertices,
                                                           const std::vector<uint32_t>& triangulated_indices) {
    const size_t num_vertices = vertices.size();
    const size_t num_triangles = triangulated_indices.size() / 3;

    // 各頂点に隣接する面法線を累積
    std::vector<glm::vec3> vertex_normals(num_vertices, glm::vec3(0.0f));
    std::vector<int> vertex_face_count(num_vertices, 0);

    for (size_t tri_idx = 0; tri_idx < num_triangles; tri_idx++) {
        const int idx0 = triangulated_indices[3 * tri_idx + 0];
        const int idx1 = triangulated_indices[3 * tri_idx + 1];
        const int idx2 = triangulated_indices[3 * tri_idx + 2];

        const glm::vec3& p0 = vertices[idx0];
        const glm::vec3& p1 = vertices[idx1];
        const glm::vec3& p2 = vertices[idx2];

        // 面法線を計算
        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 face_normal = cross(edge1, edge2);

        // 正規化せずに累積（角度による重み付け効果）
        float len = length(face_normal);
        if (len > 1e-6f) {
            face_normal /= len;

            // 各頂点に加算
            vertex_normals[idx0] += face_normal;
            vertex_normals[idx1] += face_normal;
            vertex_normals[idx2] += face_normal;

            vertex_face_count[idx0]++;
            vertex_face_count[idx1]++;
            vertex_face_count[idx2]++;
        }
    }

    // 正規化
    std::vector<glm::vec3> result;
    result.reserve(num_vertices);

    for (size_t i = 0; i < num_vertices; i++) {
        if (vertex_face_count[i] > 0) {
            glm::vec3 normal = vertex_normals[i];
            float len = length(normal);
            if (len > 1e-6f) {
                normal /= len;
            } else {
                normal = glm::vec3(0.0f, 1.0f, 0.0f);  // デフォルト
            }
            result.push_back(normal);
        } else {
            // 孤立した頂点
            result.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
        }
    }

    return result;
}

Interpolation MeshManager::token_to_interpolation(const TfToken& token) {
    if (token == HdPrimvarSchemaTokens->constant) {
        return Interpolation::Constant;
    } else if (token == HdPrimvarSchemaTokens->uniform) {
        return Interpolation::Uniform;
    } else if (token == HdPrimvarSchemaTokens->vertex) {
        return Interpolation::Vertex;
    } else if (token == HdPrimvarSchemaTokens->faceVarying) {
        return Interpolation::FaceVarying;
    } else if (token == HdPrimvarSchemaTokens->instance) {
        return Interpolation::Instance;
    }
    // デフォルトはVertex
    return Interpolation::Vertex;
}

std::vector<SdfPath> MeshManager::get_geom_subset_paths(const HdSceneIndexBaseRefPtr& scene_index,
                                                        const SdfPath& prim_path) {
    std::vector<SdfPath> subset_paths;

    // 子プリムのパスを取得
    SdfPathVector child_paths = scene_index->GetChildPrimPaths(prim_path);

    for (const SdfPath& child_path : child_paths) {
        HdSceneIndexPrim child_prim = scene_index->GetPrim(child_path);

        // GeomSubsetかどうかをチェック
        if (child_prim.primType == HdPrimTypeTokens->geomSubset) {
            subset_paths.push_back(child_path);
        }
    }

    return subset_paths;
}

MeshManager::SubsetTopologyData MeshManager::create_subset_topology(const HdMeshTopology& original_topology,
                                                                    const VtIntArray& subset_face_indices) {
    SubsetTopologyData result;

    const VtIntArray& orig_face_vertex_counts = original_topology.GetFaceVertexCounts();
    const VtIntArray& orig_face_vertex_indices = original_topology.GetFaceVertexIndices();

    // 使用される頂点を追跡
    std::unordered_map<int, int> original_to_new_vertex;
    std::vector<int> new_to_original_vertex;

    VtIntArray new_face_vertex_counts;
    VtIntArray new_face_vertex_indices;

    new_face_vertex_counts.reserve(subset_face_indices.size());

    // 元のメッシュでの頂点インデックスのオフセットを計算
    std::vector<int> face_offsets;
    face_offsets.reserve(orig_face_vertex_counts.size() + 1);
    int offset = 0;
    face_offsets.push_back(offset);
    for (int count : orig_face_vertex_counts) {
        offset += count;
        face_offsets.push_back(offset);
    }

    // サブセットの各面を処理
    for (int face_idx : subset_face_indices) {
        int face_vertex_count = orig_face_vertex_counts[face_idx];
        new_face_vertex_counts.push_back(face_vertex_count);

        // この面の頂点インデックスを取得
        int start = face_offsets[face_idx];

        for (int i = 0; i < face_vertex_count; i++) {
            int original_vertex_idx = orig_face_vertex_indices[start + i];

            // 新しい頂点インデックスを取得または作成
            auto it = original_to_new_vertex.find(original_vertex_idx);
            int new_vertex_idx;

            if (it != original_to_new_vertex.end()) {
                new_vertex_idx = it->second;
            } else {
                new_vertex_idx = new_to_original_vertex.size();
                original_to_new_vertex[original_vertex_idx] = new_vertex_idx;
                new_to_original_vertex.push_back(original_vertex_idx);
            }

            new_face_vertex_indices.push_back(new_vertex_idx);
        }
    }

    // 新しいトポロジーを構築
    result.topology = HdMeshTopology(original_topology.GetScheme(), original_topology.GetOrientation(),
                                     new_face_vertex_counts, new_face_vertex_indices);

    // 頂点マップを設定
    result.vertex_map.resize(new_to_original_vertex.size());
    for (size_t i = 0; i < new_to_original_vertex.size(); i++) {
        result.vertex_map[i] = new_to_original_vertex[i];
    }

    return result;
}
