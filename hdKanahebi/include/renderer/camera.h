#pragma once

#include <iostream>
#include <map>
#include <vector>

#include <glm/glm.hpp>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/pxr.h>

using namespace pxr;

struct CameraData {
    glm::mat4 transform;
    float fov_y;  // ラジアン単位
};

class CameraManager {
public:
    CameraManager();

    void add_camera(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);
    void delete_camera(const SdfPath& path);
    void rename_camera(const SdfPath& old_path, const SdfPath& new_path);
    void update_camera(const SdfPath& path, const HdSceneIndexBasePtr& scene_index);

    void set_main_camera_path(const SdfPath& path);

    const CameraData& get_main_camera() const;

private:
    SdfPath main_camera_path_;
    std::map<SdfPath, CameraData> cameras_;
};
