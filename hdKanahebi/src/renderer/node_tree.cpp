#include "renderer/node_tree.h"

NodeTree::NodeTree() : nodes_() {
    // ルートノードを追加
    Node root_node;
    root_node.transform = glm::mat4(1.0f);
    nodes_[SdfPath::EmptyPath()] = root_node;
}

void NodeTree::add_node(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    Node node;

    HdDataSourceBaseHandle xform_ds =
            scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("xform"), TfToken("matrix")));
    if (xform_ds) {
        HdSampledDataSourceHandle sampled_xform_ds = HdSampledDataSource::Cast(xform_ds);
        VtValue xform = sampled_xform_ds->GetValue(0);
        GfMatrix4d matrix = xform.Get<GfMatrix4d>();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                node.transform[i][j] = static_cast<float>(matrix[i][j]);
            }
        }
    } else {
        node.transform = glm::mat4(1.0f);
    }

    node.children = scene_index->GetChildPrimPaths(path);
    nodes_[path] = node;
}

void NodeTree::delete_node(const SdfPath& path) {
    auto it = nodes_.find(path);
    if (it != nodes_.end()) {
        nodes_.erase(it);
    }
}

void NodeTree::rename_node(const SdfPath& old_path, const SdfPath& new_path) {
    auto it = nodes_.find(old_path);
    if (it != nodes_.end()) {
        Node node = it->second;
        nodes_.erase(it);
        nodes_[new_path] = node;
    }
}

void NodeTree::update_node(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    auto it = nodes_.find(path);
    if (it != nodes_.end()) {
        Node& node = it->second;

        HdDataSourceBaseHandle xform_ds =
                scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("xform"), TfToken("matrix")));
        if (xform_ds) {
            HdSampledDataSourceHandle sampled_xform_ds = HdSampledDataSource::Cast(xform_ds);
            VtValue xform = sampled_xform_ds->GetValue(0.0f);
            GfMatrix4d matrix = xform.Get<GfMatrix4d>();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    node.transform[i][j] = static_cast<float>(matrix[i][j]);
                }
            }
        } else {
            node.transform = glm::mat4(1.0f);
        }

        node.children = scene_index->GetChildPrimPaths(path);
    }
}

const Node& NodeTree::get_node(const SdfPath& path) const {
    auto it = nodes_.find(path);
    if (it != nodes_.end()) {
        return it->second;
    } else {
        return nodes_.at(SdfPath::EmptyPath());
    }
}

const glm::mat4 NodeTree::get_global_transform(const SdfPath& path) const {
    glm::mat4 global_transform = glm::mat4(1.0f);

    auto it = nodes_.find(path);
    if (it != nodes_.end()) {
        const Node& node = it->second;
        global_transform = node.transform * global_transform;
    }

    return global_transform;
}
