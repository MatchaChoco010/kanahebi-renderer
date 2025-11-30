#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/tokens.h>

#include <oqmc/sobolbn.h>

#include <oxu/cuda/buffer.h>
#include <oxu/cuda/context.h>
#include <oxu/cuda/module.h>
#include <oxu/optix/context.h>
#include <oxu/optix/module.h>
#include <oxu/optix/pipeline.h>
#include <oxu/optix/program_group.h>

#include "hydra/render_buffer.h"
#include "renderer/common.h"
#include "renderer/scene.h"
#include "renderer/shared.h"

using namespace pxr;
using namespace oxu::utils;
namespace fs = std::filesystem;

class Renderer {
public:
    static Renderer create(oxu::cuda::Context& cuda_context, oxu::optix::Context& optix_context);

    Renderer(fs::path plugin_resource_path,
             oxu::cuda::Context& cuda_context,
             oxu::optix::Context& optix_context,
             oxu::cuda::Module&& cuda_module,
             oxu::optix::Module&& optix_module,
             oxu::optix::ProgramGroup&& raygen_pg,
             oxu::optix::ProgramGroup&& miss_pg,
             oxu::optix::ProgramGroup&& hit_pg,
             oxu::optix::Pipeline&& pipeline,
             oxu::cuda::Buffer&& sobol_buffer,
             oxu::cuda::Buffer&& params_buffer,
             CUfunction&& resolve_color_kernel_uint,
             CUfunction&& resolve_color_kernel_float,
             CUfunction&& resolve_depth_kernel);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    Renderer(const Renderer&&) = delete;
    Renderer&& operator=(const Renderer&&) = delete;

    /// @brief シーン更新を通知してレンダリング停止を待つ
    /// この関数呼び出しの後にGPUリソースの更新を行い、
    /// シーンの更新が完了したらend_scene_update()を呼ぶ必要がある
    void begin_scene_update();

    /// @brief シーン更新完了を通知（レンダリング再開）
    void end_scene_update();

    /// @brief ブロッキングモードを設定
    void set_blocking_mode(bool blocking) { blocking_mode_ = blocking; }

    /// @brief ターゲットspp数を設定
    void set_target_spp(int target_spp) { target_spp_ = target_spp; }

    /// @brief 収束判定を取得
    bool is_converged() const { return is_converged_.load(); }

    /// @brief 収束の進捗を取得
    float percent_complete() const {
        return 100.0f * static_cast<float>(current_spp_.load() - 1) / static_cast<float>(target_spp_);
    }

    /// @brief レンダリング深度を設定
    void set_depth(unsigned int depth) { depth_ = depth; }

    /// @brief フィルムの透明化設定
    void set_film_transparent(bool film_transparent) { film_transparent_ = film_transparent; }

    /// @brief 露出設定
    void set_exposure(float exposure) { exposure_ = exposure; }

    /// @brief Y軸反転設定
    void set_flip_y(bool flip_y) { flip_y_ = flip_y; }

    /// @brief プログラムグループを取得
    const ProgramGroups get_program_groups() const;

    /// @brief シーンデータを設定
    void set_scene_data(std::unique_ptr<const SceneData> scene_data);

    /// @brief RenderBufferを同期しレンダリング経過を書き込む
    void sync_render_buffers(const HdRenderPassAovBindingVector& bindings);

private:
    /// @brief ダブルバッファリングしたAOVのバッファ
    struct AOVBuffer {
        oxu::cuda::Buffer front_buffer;
        oxu::cuda::Buffer back_buffer;
        HdFormat format;
        size_t bytes_per_pixel;
        int width;
        int height;

        void resize(int w, int h, HdFormat fmt, size_t bytes_per_pixel) {
            width = w;
            height = h;
            format = fmt;
            this->bytes_per_pixel = bytes_per_pixel;
            size_t size = w * h * bytes_per_pixel;
            front_buffer = oxu::cuda::Buffer(size);
            back_buffer = oxu::cuda::Buffer(size);
        }

        void clear() {
            front_buffer.memset(0);
            back_buffer.memset(0);
        }
    };

    /// @brief AOV情報
    struct AovInfo {
        TfToken name;
        HdFormat format;
        int width;
        int height;

        bool operator==(const AovInfo& other) const {
            return name == other.name && format == other.format && width == other.width && height == other.height;
        }
    };

    /// @brief バッファをクリア
    void clear_buffers();

    /// @brief HdRenderPassAovBindingVectorからAovInfoリストを抽出
    std::vector<AovInfo> extract_aov_infos(const HdRenderPassAovBindingVector& bindings) const;

    /// @brief AOV構成の変更を検出
    bool detect_aov_changes(const std::vector<AovInfo>& new_infos) const;

    /// @brief AOVバッファを更新
    void update_aov_buffers(const std::vector<AovInfo>& aov_infos);

    /// @brief RenderBufferへデータを書き込む
    void write_to_render_buffers(const HdRenderPassAovBindingVector& bindings) const;

    /// @brief 更新準備完了を通知して完了を待つ（レンダリングスレッド内部用）
    void yield_for_external_update();

    /// @brief 収束を同期的に待機する
    void wait_for_convergence();

    /// @brief レンダリングスレッドのメインループ
    void render_thread_main();

    /// @brief render関数
    void render(const unsigned int spp);

    fs::path plugin_resource_path_;

    unsigned int target_spp_{64};
    std::atomic<unsigned int> current_spp_{1};
    unsigned int depth_{16};
    bool blocking_mode_{false};
    bool film_transparent_{false};
    float exposure_{1.0f};
    bool flip_y_{false};

    oxu::cuda::Context& cuda_context_;
    oxu::optix::Context& optix_context_;
    oxu::cuda::Module cuda_module_;
    oxu::optix::Module optix_module_;
    oxu::optix::ProgramGroup raygen_pg_;
    oxu::optix::ProgramGroup miss_pg_;
    oxu::optix::ProgramGroup hit_pg_;
    oxu::optix::Pipeline pipeline_;
    oxu::cuda::Buffer params_buffer_;

    std::unique_ptr<const SceneData> scene_data_;
    oxu::cuda::Buffer sobol_buffer_;
    CUfunction resolve_color_kernel_uint_;
    CUfunction resolve_color_kernel_float_;
    CUfunction resolve_depth_kernel_;

    std::map<TfToken, AOVBuffer> aov_buffers_;
    std::vector<AovInfo> current_aov_infos_;

    unsigned int width_{0};
    unsigned int height_{0};
    oxu::cuda::Buffer color_buffer_;
    oxu::cuda::Buffer depth_buffer_;

    // レンダリング状態
    std::atomic<bool> is_converged_{true};
    std::atomic<bool> scene_update_required_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_rendering_{false};

    // スレッド制御
    std::thread render_thread_;

    // レンダリング開始/停止用
    std::mutex render_state_mutex_;
    std::condition_variable render_state_cv_;

    // 更新通知用
    std::mutex update_mutex_;
    std::condition_variable update_cv_;

    // バッファアクセス用
    mutable std::mutex buffer_mutex_;

    // 収束通知用
    std::mutex converge_mutex_;
    std::condition_variable converge_cv_;
};
