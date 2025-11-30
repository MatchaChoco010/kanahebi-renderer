#pragma once

#include "math/matrix.h"
#include "math/vector.h"

#include "params.h"

__device__ void generate_camera_ray(const uint2& idx,
                                    const uint2& dim,
                                    float3& ray_origin,
                                    float3& ray_direction,
                                    const float random[2]) {
    SceneParams* scene_params = reinterpret_cast<SceneParams*>(params.scene_params);
    CameraParams& camera_params = scene_params->camera_params;

    ray_origin = make_float3(camera_params.transform.rows[0].w, camera_params.transform.rows[1].w,
                             camera_params.transform.rows[2].w);

    float aspect_ratio = static_cast<float>(dim.x) / static_cast<float>(dim.y);

    float px = (static_cast<float>(idx.x) + random[0]) / static_cast<float>(dim.x);
    float py = (static_cast<float>(idx.y) + random[1]) / static_cast<float>(dim.y);

    float screen_x = (2.0f * px - 1.0f) * aspect_ratio * tanf(camera_params.fov_y * 0.5f);
    float screen_y = (2.0f * py - 1.0f) * tanf(camera_params.fov_y * 0.5f);
    float3 ray_dir_camera = normalize(make_float3(screen_x, screen_y, -1.0f));
    float4 ray_dir_world =
            camera_params.transform * make_float4(ray_dir_camera.x, ray_dir_camera.y, ray_dir_camera.z, 0.0f);
    ray_direction = normalize(make_float3(ray_dir_world.x, ray_dir_world.y, ray_dir_world.z));
}
