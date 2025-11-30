#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include <oxu/cuda/texture.h>

#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/materialConnectionSchema.h>
#include <pxr/imaging/hd/materialNetworkSchema.h>
#include <pxr/imaging/hd/materialNodeParameterSchema.h>
#include <pxr/imaging/hd/materialNodeSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include "renderer/common.h"
#include "renderer/shared.h"

using namespace pxr;

struct MaterialData {
    SdfPath material_path;
    glm::vec3 base_color;
    float metallic;
    float roughness;
    glm::vec3 emissive_color;
    float opacity;
    float ior;

    std::unordered_map<std ::string, oxu::cuda::Texture> textures;
};

class MaterialManager {
public:
    MaterialManager();

    void add_material(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);
    void delete_material(const SdfPath& path);
    const MaterialData& get_material(const SdfPath& prim_path) const;
    const BuildMaterialResult build_materials() const;

private:
    std::map<SdfPath, MaterialData> materials_;
};
