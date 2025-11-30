#include "renderer/material.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

MaterialManager::MaterialManager() {
    MaterialData default_material;
    default_material.material_path = SdfPath::EmptyPath();
    default_material.base_color = glm::vec3(0.0f, 0.0f, 0.0f);
    default_material.metallic = 0.0f;
    default_material.roughness = 0.5f;
    default_material.emissive_color = glm::vec3(1.0f, 0.0f, 1.0f);
    default_material.opacity = 1.0f;
    default_material.ior = 1.5f;
    materials_.emplace(default_material.material_path, std::move(default_material));
}

void MaterialManager::add_material(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    HdSceneIndexPrim prim = scene_index->GetPrim(path);
    if (!prim.dataSource) {
        return;
    }

    // material schema を取得
    HdMaterialSchema matSchema = HdMaterialSchema::GetFromParent(prim.dataSource);
    if (!matSchema.IsDefined()) {
        return;
    }

    // MaterialNetwork を取得
    HdMaterialNetworkSchema netSchema = matSchema.GetMaterialNetwork();

    // ターミナル接続を取得
    HdMaterialConnectionContainerSchema terminals = netSchema.GetTerminals();
    HdMaterialConnectionSchema surfConn = terminals.Get(TfToken("surface"));
    if (!surfConn.IsDefined()) {
        return;
    }

    // 上流ノードの名前を取得（nodePath トークン）
    HdTokenDataSourceHandle nodePathDs = surfConn.GetUpstreamNodePath();
    if (!nodePathDs) {
        return;
    }
    TfToken nodePath = nodePathDs->GetTypedValue(0.0f);

    // ノード群から該当ノードを取得（locator経由）
    HdMaterialNodeContainerSchema nodes = netSchema.GetNodes();
    HdMaterialNodeSchema surfaceNode = nodes.Get(nodePath);
    if (!surfaceNode.IsDefined()) {
        return;
    }

    // シェーダIDの確認（UsdPreviewSurface 以外はスキップ）
    if (HdTokenDataSourceHandle idDs = surfaceNode.GetNodeIdentifier()) {
        const TfToken shaderId = idDs->GetTypedValue(0.0f);
        if (shaderId != TfToken("UsdPreviewSurface")) {
            std::cout << "Unsupported shader ID: " << shaderId.GetString() << " at path: " << path.GetString()
                      << std::endl;
            return;
        }
    } else {
        return;
    }

    MaterialData material;
    material.material_path = path;

    // 値パラメータを取得
    HdMaterialNodeParameterContainerSchema params = surfaceNode.GetParameters();
    auto getFloat = [&](const char* name, float def) -> float {
        HdMaterialNodeParameterSchema p = params.Get(TfToken(name));
        if (!p.IsDefined())
            return def;
        if (HdFloatDataSourceHandle v = HdFloatDataSource::Cast(p.GetValue())) {
            return v->GetTypedValue(0.0f);
        }
        if (HdSampledDataSourceHandle sd = HdSampledDataSource::Cast(p.GetValue())) {
            VtValue val = sd->GetValue(0.0f);
            if (val.IsHolding<float>()) {
                return val.UncheckedGet<float>();
            }
        }

        return def;
    };
    auto getVec3 = [&](const char* name, const glm::vec3& def) -> glm::vec3 {
        HdMaterialNodeParameterSchema p = params.Get(TfToken(name));
        if (!p.IsDefined())
            return def;
        if (HdVec3fDataSourceHandle v = HdVec3fDataSource::Cast(p.GetValue())) {
            const GfVec3f gv = v->GetTypedValue(0.0f);
            return glm::vec3(gv[0], gv[1], gv[2]);
        }
        if (HdSampledDataSourceHandle sd = HdSampledDataSource::Cast(p.GetValue())) {
            VtValue val = sd->GetValue(0.0f);
            if (val.IsHolding<GfVec3f>()) {
                const GfVec3f gv = val.UncheckedGet<GfVec3f>();
                return glm::vec3(gv[0], gv[1], gv[2]);
            }
        }
        return def;
    };

    material.base_color = getVec3("diffuseColor", glm::vec3(1.0f));
    material.metallic = getFloat("metallic", 0.0f);
    material.roughness = getFloat("roughness", 0.0f);
    material.emissive_color = getVec3("emissiveColor", glm::vec3(0.0f));
    material.opacity = getFloat("opacity", 1.0f);
    material.ior = getFloat("ior", 1.5f);

    // テクスチャ接続を処理
    HdMaterialConnectionVectorContainerSchema conns = surfaceNode.GetInputConnections();

    auto loadTextureFromNodePath = [&](const TfToken& texNodePath, const std::string& slotName) {
        HdMaterialNodeSchema texNode = nodes.Get(texNodePath);
        if (!texNode.IsDefined())
            return;

        // UsdUVTextureを想定
        HdMaterialNodeParameterContainerSchema tparams = texNode.GetParameters();
        HdMaterialNodeParameterSchema fileParam = tparams.Get(TfToken("file"));
        if (!fileParam.IsDefined())
            return;

        std::string filePath;
        if (HdAssetPathDataSourceHandle ds = HdAssetPathDataSource::Cast(fileParam.GetValue())) {
            const SdfAssetPath ap = ds->GetTypedValue(0.0f);
            filePath = ap.GetResolvedPath().empty() ? ap.GetAssetPath() : ap.GetResolvedPath();
        } else if (HdStringDataSourceHandle sds = HdStringDataSource::Cast(fileParam.GetValue())) {
            filePath = sds->GetTypedValue(0.0f);
        }

        if (filePath.empty())
            return;

        int w = 0, h = 0, ch = 0;
        unsigned char* pixels = stbi_load(filePath.c_str(), &w, &h, &ch, 4);
        if (!pixels) {
            return;
        }

        oxu::cuda::Texture tex(pixels, w, h, 4);
        stbi_image_free(pixels);

        material.textures.emplace(slotName, std::move(tex));
    };

    // Surface から直接ぶら下がっている典型的な入力名
    const char* kTexInputs[] = {"diffuseColor", "emissiveColor", "normal", "metallic", "roughness"};
    for (const char* input : kTexInputs) {
        auto vecSchema = conns.Get(TfToken(input));
        if (!vecSchema.IsDefined())
            continue;

        // 最初の接続だけ見る
        HdMaterialConnectionSchema conn = vecSchema.GetElement(0);
        if (!conn.IsDefined())
            continue;

        if (HdTokenDataSourceHandle upDs = conn.GetUpstreamNodePath()) {
            const TfToken texNodePath = upDs->GetTypedValue(0.0f);
            loadTextureFromNodePath(texNodePath, input);
        }
    }

    materials_.emplace(path, std::move(material));
}

void MaterialManager::delete_material(const SdfPath& path) {
    if (materials_.find(path) != materials_.end()) {
        materials_.erase(path);
    }
}

const MaterialData& MaterialManager::get_material(const SdfPath& prim_path) const {
    return materials_.at(prim_path);
}

const BuildMaterialResult MaterialManager::build_materials() const {
    BuildMaterialResult result;

    std::vector<MaterialParams> material_data_list;

    uint32_t index = 0;
    for (const auto& [prim_path, material] : materials_) {
        result.material_index_map[prim_path] = index++;
        MaterialParams material_params{};
        material_params.base_color = make_float3(material.base_color.r, material.base_color.g, material.base_color.b);
        material_params.metallic = material.metallic;
        material_params.roughness = material.roughness;
        material_params.emissive_color =
                make_float3(material.emissive_color.r, material.emissive_color.g, material.emissive_color.b);
        material_params.opacity = material.opacity;
        material_params.ior = material.ior;

        material_params.has_base_color_texture = material.textures.find("diffuseColor") != material.textures.end();
        material_params.has_normal_texture = material.textures.find("normal") != material.textures.end();
        material_params.has_metallic_texture = material.textures.find("metallic") != material.textures.end();
        material_params.has_roughness_texture = material.textures.find("roughness") != material.textures.end();
        material_params.has_emissive_color_texture = material.textures.find("emissiveColor") != material.textures.end();

        if (material_params.has_base_color_texture) {
            material_params.base_color_texture = material.textures.at("diffuseColor").handle();
        }
        if (material_params.has_normal_texture) {
            material_params.normal_texture = material.textures.at("normal").handle();
        }
        if (material_params.has_metallic_texture) {
            material_params.metallic_texture = material.textures.at("metallic").handle();
        }
        if (material_params.has_roughness_texture) {
            material_params.roughness_texture = material.textures.at("roughness").handle();
        }
        if (material_params.has_emissive_color_texture) {
            material_params.emissive_color_texture = material.textures.at("emissiveColor").handle();
        }

        material_data_list.emplace_back(material_params);
    }

    result.material_list_buffer = oxu::cuda::Buffer(material_data_list.size() * sizeof(MaterialParams));
    result.material_list_buffer.upload(
            std::span<const MaterialParams>(material_data_list.data(), material_data_list.size()));

    return result;
}
