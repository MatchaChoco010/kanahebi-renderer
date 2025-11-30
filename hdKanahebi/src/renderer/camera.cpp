#include "renderer/camera.h"

CameraManager::CameraManager() : cameras_(), main_camera_path_(SdfPath::EmptyPath()) {
    cameras_[SdfPath::EmptyPath()] = CameraData{glm::mat4(1.0f), glm::radians(60.0f)};
}

void CameraManager::add_camera(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    CameraData camera_data;

    HdDataSourceBaseHandle xform_ds =
            scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("xform"), TfToken("matrix")));
    if (xform_ds) {
        HdSampledDataSourceHandle sampled_xform_ds = HdSampledDataSource::Cast(xform_ds);
        VtValue xform = sampled_xform_ds->GetValue(0.0f);
        GfMatrix4d matrix = xform.Get<GfMatrix4d>();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                camera_data.transform[i][j] = static_cast<float>(matrix[i][j]);
            }
        }
    } else {
        camera_data.transform = glm::mat4(1.0f);
    }

    HdDataSourceBaseHandle vertical_aperture_ds =
            scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("camera"), TfToken("verticalAperture")));
    HdDataSourceBaseHandle focal_length_ds =
            scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("camera"), TfToken("focalLength")));
    if (vertical_aperture_ds && focal_length_ds) {
        HdSampledDataSourceHandle sampled_vertical_aperture_ds = HdSampledDataSource::Cast(vertical_aperture_ds);
        VtValue vertical_aperture_value = sampled_vertical_aperture_ds->GetValue(0.0f);
        float vertical_aperture = vertical_aperture_value.Get<float>();

        HdSampledDataSourceHandle sampled_focal_length_ds = HdSampledDataSource::Cast(focal_length_ds);
        VtValue focal_length_value = sampled_focal_length_ds->GetValue(0.0f);
        float focal_length = focal_length_value.Get<float>();

        camera_data.fov_y = 2.0f * atan(vertical_aperture / (2.0f * focal_length));
    } else {
        camera_data.fov_y = glm::radians(60.0f);  // デフォルト値
    }

    cameras_[path] = camera_data;
}

void CameraManager::delete_camera(const SdfPath& path) {
    auto it = cameras_.find(path);
    if (it != cameras_.end()) {
        cameras_.erase(it);
    }
}

void CameraManager::rename_camera(const SdfPath& old_path, const SdfPath& new_path) {
    auto it = cameras_.find(old_path);
    if (it != cameras_.end()) {
        CameraData camera_data = it->second;
        cameras_.erase(it);
        cameras_[new_path] = camera_data;
    }
}

void CameraManager::update_camera(const SdfPath& path, const HdSceneIndexBasePtr& scene_index) {
    auto it = cameras_.find(path);
    if (it != cameras_.end()) {
        CameraData& camera_data = it->second;

        HdDataSourceBaseHandle xform_ds =
                scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("xform"), TfToken("matrix")));
        if (xform_ds) {
            HdSampledDataSourceHandle sampled_xform_ds = HdSampledDataSource::Cast(xform_ds);
            VtValue xform = sampled_xform_ds->GetValue(0.0f);
            GfMatrix4d matrix = xform.Get<GfMatrix4d>();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    camera_data.transform[i][j] = static_cast<float>(matrix[i][j]);
                }
            }
        }

        HdDataSourceBaseHandle vertical_aperture_ds =
                scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("camera"), TfToken("verticalAperture")));
        HdDataSourceBaseHandle focal_length_ds =
                scene_index->GetDataSource(path, HdDataSourceLocator(TfToken("camera"), TfToken("focalLength")));
        if (vertical_aperture_ds && focal_length_ds) {
            HdSampledDataSourceHandle sampled_vertical_aperture_ds = HdSampledDataSource::Cast(vertical_aperture_ds);
            VtValue vertical_aperture_value = sampled_vertical_aperture_ds->GetValue(0.0f);
            float vertical_aperture = vertical_aperture_value.Get<float>();

            HdSampledDataSourceHandle sampled_focal_length_ds = HdSampledDataSource::Cast(focal_length_ds);
            VtValue focal_length_value = sampled_focal_length_ds->GetValue(0.0f);
            float focal_length = focal_length_value.Get<float>();

            camera_data.fov_y = 2.0f * atan(vertical_aperture / (2.0f * focal_length));
        }
    }
}

void CameraManager::set_main_camera_path(const SdfPath& path) {
    main_camera_path_ = path;
}

const CameraData& CameraManager::get_main_camera() const {
    auto it = cameras_.find(main_camera_path_);
    if (it != cameras_.end()) {
        return it->second;
    } else {
        return cameras_.at(SdfPath::EmptyPath());
    }
}
