#pragma once

#include <map>
#include <vector>

#include <glm/glm.hpp>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/pxr.h>

using namespace pxr;

struct Node {
    glm::mat4 transform;
    std::vector<SdfPath> children;
};

class NodeTree {
public:
    NodeTree();

    void add_node(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);
    void delete_node(const SdfPath& path);
    void rename_node(const SdfPath& old_path, const SdfPath& new_path);
    void update_node(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);
    const Node& get_node(const SdfPath& path) const;
    const glm::mat4 get_global_transform(const SdfPath& path) const;

private:
    std::map<SdfPath, Node> nodes_;
};
