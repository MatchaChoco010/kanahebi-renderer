#include "renderer/renderer.h"

Renderer Renderer::create(oxu::cuda::Context& cuda_context, oxu::optix::Context& optix_context) {
    // Get plugin resource path
    pxr::PlugPluginPtr plugin = pxr::PlugRegistry::GetInstance().GetPluginWithName("hdKanahebi");
    fs::path plugin_resource_path = fs::path(plugin->GetResourcePath());

    // --- Pipeline options ---
    OptixPipelineCompileOptions pcomp = {};
    pcomp.usesMotionBlur = false;
    pcomp.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    // pcomp.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pcomp.numPayloadValues = 2;
    pcomp.numAttributeValues = 2;
    pcomp.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pcomp.pipelineLaunchParamsVariableName = "params";

    // --- Module ---
    const fs::path ptx_path = plugin_resource_path / "ptx/render.ptx";
    oxu::optix::Module optix_module(optix_context, pcomp, ptx_path);

    // --- Program groups ---
    OptixProgramGroupOptions pg_opt = {};

    OptixProgramGroupDesc rg_desc = {};
    rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_desc.raygen.module = optix_module;
    rg_desc.raygen.entryFunctionName = "__raygen__rg";
    oxu::optix::ProgramGroup raygen_pg(optix_context, rg_desc, pg_opt);

    OptixProgramGroupDesc ms_desc = {};
    ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_desc.miss.module = optix_module;
    ms_desc.miss.entryFunctionName = "__miss__ms";
    oxu::optix::ProgramGroup miss_pg(optix_context, ms_desc, pg_opt);

    OptixProgramGroupDesc hg_desc = {};
    hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg_desc.hitgroup.moduleCH = optix_module;
    hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    oxu::optix::ProgramGroup hit_pg(optix_context, hg_desc, pg_opt);

    // --- Pipeline ---
    OptixPipelineLinkOptions linkopt = {};

    OptixProgramGroup groups[] = {raygen_pg, miss_pg, hit_pg};

    uint32_t continuation_stack_size = 2048;  // TODO: きちんと計算する
    uint32_t max_traversable_graph_depth = 2;
    oxu::optix::Pipeline pipeline(optix_context, &pcomp, &linkopt, groups, continuation_stack_size,
                                  max_traversable_graph_depth);

    // --- Sobol buffer ---
    char sobol_data[oqmc::SobolBnSampler::cacheSize];
    oqmc::SobolBnSampler::initialiseCache(sobol_data);
    oxu::cuda::Buffer sobol_buffer(sizeof(sobol_data));
    sobol_buffer.upload(std::span<const char>(sobol_data, sizeof(sobol_data)));

    // --- Params Buffer ---
    oxu::cuda::Buffer params_buffer(sizeof(Params));

    // --- Resolve Kernel ---
    fs::path resolve_ptx_path = plugin_resource_path / "ptx/resolve.ptx";
    oxu::cuda::Module cuda_module(cuda_context, resolve_ptx_path);
    CUfunction resolve_color_kernel_uint = cuda_module.get_function("resolve_color_kernel_uint");
    CUfunction resolve_color_kernel_float = cuda_module.get_function("resolve_color_kernel_float");
    CUfunction resolve_depth_kernel = cuda_module.get_function("resolve_depth_kernel");

    return Renderer(std::move(plugin_resource_path), cuda_context, optix_context, std::move(cuda_module),
                    std::move(optix_module), std::move(raygen_pg), std::move(miss_pg), std::move(hit_pg),
                    std::move(pipeline), std::move(sobol_buffer), std::move(params_buffer),
                    std::move(resolve_color_kernel_uint), std::move(resolve_color_kernel_float),
                    std::move(resolve_depth_kernel));
}

Renderer::Renderer(fs::path plugin_resource_path,
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
                   CUfunction&& resolve_depth_kernel)
    : plugin_resource_path_(std::move(plugin_resource_path)),
      cuda_context_(cuda_context),
      optix_context_(optix_context),
      cuda_module_(std::move(cuda_module)),
      optix_module_(std::move(optix_module)),
      raygen_pg_(std::move(raygen_pg)),
      miss_pg_(std::move(miss_pg)),
      hit_pg_(std::move(hit_pg)),
      pipeline_(std::move(pipeline)),
      sobol_buffer_(std::move(sobol_buffer)),
      params_buffer_(std::move(params_buffer)),
      resolve_color_kernel_uint_(std::move(resolve_color_kernel_uint)),
      resolve_color_kernel_float_(std::move(resolve_color_kernel_float)),
      resolve_depth_kernel_(std::move(resolve_depth_kernel)) {
    render_thread_ = std::thread(&Renderer::render_thread_main, this);
}

Renderer::~Renderer() {
    should_stop_.store(true);
    scene_update_required_.store(false);

    render_state_cv_.notify_all();
    update_cv_.notify_all();
    converge_cv_.notify_all();

    if (render_thread_.joinable()) {
        render_thread_.join();
    }
}

void Renderer::begin_scene_update() {
    scene_update_required_.store(true);
    is_converged_.store(false);

    // レンダリングスレッドを起こす
    render_state_cv_.notify_one();

    // レンダリングが停止するまで待つ
    std::unique_lock<std::mutex> lock(update_mutex_);
    update_cv_.wait(lock, [this] { return !is_rendering_.load(); });
}

void Renderer::end_scene_update() {
    scene_update_required_.store(false);

    std::lock_guard<std::mutex> lock(update_mutex_);
    update_cv_.notify_all();
}

const ProgramGroups Renderer::get_program_groups() const {
    // return ProgramGroups{raygen_pg_, {&miss_pg_, &miss_pg_}, {&hit_pg_, &hit_pg_}};
    return ProgramGroups{raygen_pg_, {&miss_pg_}, {&hit_pg_}};
}

void Renderer::set_scene_data(std::unique_ptr<const SceneData> scene_data) {
    scene_data_ = std::move(scene_data);
}

void Renderer::sync_render_buffers(const HdRenderPassAovBindingVector& bindings) {
    // AOV構成の変更を検出
    std::vector<AovInfo> new_aov_infos = extract_aov_infos(bindings);

    if (detect_aov_changes(new_aov_infos)) {
        // レンダリング停止を待機
        begin_scene_update();

        // バッファを更新
        update_aov_buffers(new_aov_infos);
        current_aov_infos_ = new_aov_infos;

        // 更新完了を通知
        end_scene_update();
    }

    // ブロッキングモードなら収束を待つ
    if (blocking_mode_) {
        wait_for_convergence();
    }

    // データをコピー
    write_to_render_buffers(bindings);
}

void Renderer::clear_buffers() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    cuda_context_.make_current();

    for (auto& [aovName, buffer] : aov_buffers_) {
        buffer.clear();
    }

    float alpha = film_transparent_ ? 0.0f : 1.0f;
    std::vector<float4> clear_color(width_ * height_, make_float4(0.0f, 0.0f, 0.0f, alpha));
    color_buffer_.upload(std::span<const float4>(clear_color.data(), clear_color.size()));
    depth_buffer_.memset(1.0f);
}

std::vector<Renderer::AovInfo> Renderer::extract_aov_infos(const HdRenderPassAovBindingVector& bindings) const {
    std::vector<AovInfo> infos;

    for (const auto& binding : bindings) {
        if (!binding.renderBuffer) {
            continue;
        }

        HdRenderBuffer* rb = static_cast<HdRenderBuffer*>(binding.renderBuffer);

        AovInfo info;
        info.name = binding.aovName;
        info.format = rb->GetFormat();
        info.width = rb->GetWidth();
        info.height = rb->GetHeight();

        infos.push_back(info);
    }

    return infos;
}

bool Renderer::detect_aov_changes(const std::vector<AovInfo>& new_infos) const {
    if (current_aov_infos_.size() != new_infos.size()) {
        return true;
    }

    for (const auto& new_info : new_infos) {
        bool found = false;
        for (const auto& current_info : current_aov_infos_) {
            if (new_info == current_info) {
                found = true;
                break;
            }
        }
        if (!found) {
            return true;
        }
    }

    return false;
}

void Renderer::update_aov_buffers(const std::vector<AovInfo>& aov_infos) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    cuda_context_.make_current();

    std::set<TfToken> requested_aov_names;
    for (const auto& info : aov_infos) {
        requested_aov_names.insert(info.name);
    }

    // 不要なAOVを削除
    for (auto it = aov_buffers_.begin(); it != aov_buffers_.end();) {
        if (requested_aov_names.find(it->first) == requested_aov_names.end()) {
            it = aov_buffers_.erase(it);
        } else {
            ++it;
        }
    }

    // AOVを追加/更新
    for (const AovInfo& info : aov_infos) {
        auto it = aov_buffers_.find(info.name);

        if (it != aov_buffers_.end()) {
            AOVBuffer& buffer = it->second;

            // 既存のAOVを更新
            if (buffer.format != info.format || buffer.width != info.width || buffer.height != info.height) {
                size_t bytes_per_pixel = HdDataSizeOfFormat(info.format);
                buffer.resize(info.width, info.height, info.format, bytes_per_pixel);
            }
        } else {
            // 新しいAOVを追加
            AOVBuffer buffer;
            size_t bytes_per_pixel = HdDataSizeOfFormat(info.format);
            buffer.resize(info.width, info.height, info.format, bytes_per_pixel);

            aov_buffers_[info.name] = std::move(buffer);
        }

        // レンダリング用バッファサイズを更新
        if (info.width != width_ || info.height != height_) {
            width_ = info.width;
            height_ = info.height;

            color_buffer_ = oxu::cuda::Buffer(width_ * height_ * sizeof(float4));
            depth_buffer_ = oxu::cuda::Buffer(width_ * height_ * sizeof(float));
        }
    }
}

void Renderer::write_to_render_buffers(const HdRenderPassAovBindingVector& bindings) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    cuda_context_.make_current();

    for (const auto& binding : bindings) {
        if (!binding.renderBuffer) {
            continue;
        }

        auto it = aov_buffers_.find(binding.aovName);
        if (it == aov_buffers_.end()) {
            continue;
        }

        const AOVBuffer& buffer = it->second;

        HdKanahebiRenderBuffer* rb = dynamic_cast<HdKanahebiRenderBuffer*>(binding.renderBuffer);
        if (rb) {
            rb->update_data(buffer.front_buffer, is_converged_.load());
        }
    }
}

void Renderer::yield_for_external_update() {
    // 更新可能になったことを通知
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        update_cv_.notify_all();
    }

    // 更新完了を待つ
    {
        std::unique_lock<std::mutex> lock(update_mutex_);
        update_cv_.wait(lock, [this] { return !scene_update_required_.load(); });
    }
}

void Renderer::wait_for_convergence() {
    std::unique_lock<std::mutex> lock(converge_mutex_);
    converge_cv_.wait(lock, [this] { return is_converged_.load(); });
}

void Renderer::render_thread_main() {
    while (!should_stop_.load()) {
        // 待機状態
        is_rendering_.store(false);

        {
            std::unique_lock<std::mutex> lock(render_state_mutex_);
            render_state_cv_.wait(lock, [this] {
                return !is_converged_.load() || scene_update_required_.load() || should_stop_.load();
            });
        }

        if (should_stop_.load()) {
            break;
        }

        // レンダリング開始前処理: バッファクリアとリセット
        current_spp_.store(1);
        clear_buffers();
        is_rendering_.store(true);

        // レンダリングループ
        while (current_spp_.load() <= target_spp_) {
            if (should_stop_.load()) {
                break;
            }

            // シーン更新チェック
            if (scene_update_required_.load()) {
                break;
            }

            // 各AOVをレンダリング
            render(current_spp_.load());

            // バッファスワップ
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);

                for (auto& [aovName, buffer] : aov_buffers_) {
                    std::swap(buffer.front_buffer, buffer.back_buffer);
                }
            }

            current_spp_.fetch_add(1);
        }

        is_rendering_.store(false);

        // シーン更新の場合
        if (scene_update_required_.load()) {
            // 準備完了を通知して完了を待つ
            yield_for_external_update();

            continue;
        }

        // 収束判定
        if (current_spp_.load() >= target_spp_) {
            is_converged_.store(true);

            {
                std::lock_guard<std::mutex> lock(converge_mutex_);
                converge_cv_.notify_all();
            }
        }
    }
}

void Renderer::render(const unsigned int spp) {
    if (width_ == 0 || height_ == 0) {
        return;
    }

    if (scene_data_ == nullptr) {
        return;
    }

    // Launch OptiX kernel
    Params params{color_buffer_.device_ptr(),
                  depth_buffer_.device_ptr(),
                  width_,
                  height_,
                  spp,
                  depth_,
                  film_transparent_ ? 1u : 0u,
                  exposure_,
                  scene_data_->scene_params_buffer.device_ptr(),
                  sobol_buffer_.device_ptr()};
    params_buffer_.upload(std::span<const Params>(&params, 1));
    optix_check(optixLaunch(pipeline_, /*stream*/ 0, params_buffer_.device_ptr(), sizeof(Params), &scene_data_->sbt,
                            width_, height_, 1));

    // Launch resolve kernel
    unsigned int N = spp;
    CUdeviceptr color_buffer_ptr = color_buffer_.device_ptr();
    CUdeviceptr depth_buffer_ptr = depth_buffer_.device_ptr();

    bool resolve_color = false;
    bool float_color = false;
    CUdeviceptr out_color_buffer_ptr;
    for (const auto& [aov_name, buffer] : aov_buffers_) {
        if (aov_name == HdAovTokens->color) {
            out_color_buffer_ptr = buffer.back_buffer.device_ptr();
            if (buffer.format == HdFormatFloat32Vec4) {
                float_color = true;
            } else {
                float_color = false;
            }
            resolve_color = true;
        }
    }
    if (resolve_color) {
        void* args[] = {&color_buffer_ptr, &out_color_buffer_ptr, &width_, &height_, &N, &flip_y_};
        if (float_color) {
            cu_check(cuLaunchKernel(
                    /*kernel*/ resolve_color_kernel_float_,
                    /*grid dim*/ (width_ + 15) / 16, (height_ + 15) / 16, 1,
                    /*block dim*/ 16, 16, 1,
                    /*shared mem*/ 0,
                    /*stream*/ 0,
                    /*arguments*/ args, nullptr));
        } else {
            cu_check(cuLaunchKernel(
                    /*kernel*/ resolve_color_kernel_uint_,
                    /*grid dim*/ (width_ + 15) / 16, (height_ + 15) / 16, 1,
                    /*block dim*/ 16, 16, 1,
                    /*shared mem*/ 0,
                    /*stream*/ 0,
                    /*arguments*/ args, nullptr));
        }
    }

    bool resolve_depth = false;
    CUdeviceptr out_depth_buffer_ptr;
    for (const auto& [aov_name, buffer] : aov_buffers_) {
        if (aov_name == HdAovTokens->depth) {
            out_depth_buffer_ptr = buffer.back_buffer.device_ptr();
            resolve_depth = true;
        }
    }
    if (resolve_depth) {
        void* args[] = {&depth_buffer_ptr, &out_depth_buffer_ptr, &width_, &height_, &N, &flip_y_};
        cu_check(cuLaunchKernel(
                /*kernel*/ resolve_depth_kernel_,
                /*grid dim*/ (width_ + 15) / 16, (height_ + 15) / 16, 1,
                /*block dim*/ 16, 16, 1,
                /*shared mem*/ 0,
                /*stream*/ 0,
                /*arguments*/ args, nullptr));
    }

    // Synchronize CUDA device
    cuda_check(cudaDeviceSynchronize());
}
