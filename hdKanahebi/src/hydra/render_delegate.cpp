#include "hydra/render_delegate.h"

PXR_NAMESPACE_OPEN_SCOPE

HdKanahebiRenderDelegate::HdKanahebiRenderDelegate()
    : HdRenderDelegate(),
      cuda_context_(),
      optix_context_(cuda_context_),
      renderer_(Renderer::create(cuda_context_, optix_context_)),
      scene_(optix_context_) {
    initialize();
}

HdKanahebiRenderDelegate::HdKanahebiRenderDelegate(HdRenderSettingsMap const& settingsMap)
    : HdRenderDelegate(settingsMap),
      cuda_context_(),
      optix_context_(cuda_context_),
      renderer_(Renderer::create(cuda_context_, optix_context_)),
      scene_(optix_context_) {
    initialize();

    for (const auto& setting : settingsMap) {
        SetRenderSetting(setting.first, setting.second);
    }
}

HdKanahebiRenderDelegate::~HdKanahebiRenderDelegate() {
    // Cleanup scene index observer
    if (scene_index_) {
        scene_index_->RemoveObserver(HdSceneIndexObserverPtr(&scene_index_observer_));
    }

    // Cleanup resource registry
    resource_registry_.reset();

    std::cout << "HdKanahebiRenderDelegate destroyed." << std::endl;
}

const TfTokenVector& HdKanahebiRenderDelegate::GetSupportedRprimTypes() const {
    // return empty
    static const TfTokenVector supported_rprim_types;
    return supported_rprim_types;
}

const TfTokenVector& HdKanahebiRenderDelegate::GetSupportedSprimTypes() const {
#ifndef IS_BLENDER_4_5
    // return empty
    static const TfTokenVector supported_sprim_types;
#else
    // Blenderはカメラのsprimの対応を明示しないとSceneIndexにカメラが登録されない
    static const TfTokenVector supported_sprim_types{HdPrimTypeTokens->camera};
#endif
    return supported_sprim_types;
}

const TfTokenVector& HdKanahebiRenderDelegate::GetSupportedBprimTypes() const {
    static const TfTokenVector supported_bprim_types{HdPrimTypeTokens->renderBuffer};
    return supported_bprim_types;
}

HdResourceRegistrySharedPtr HdKanahebiRenderDelegate::GetResourceRegistry() const {
    return resource_registry_;
}

void HdKanahebiRenderDelegate::SetRenderSetting(TfToken const& key, VtValue const& value) {
    if (key == TfToken("renderCameraPath")) {
        SdfPath camera_path = value.Get<SdfPath>();
        scene_.set_main_camera_path(camera_path);
    } else if (key == TfToken("kanahebi:global:targetsamples")) {
        if (value.IsHolding<long>()) {
            long samples = value.Get<long>();
            renderer_.set_target_spp(samples);
        } else if (value.IsHolding<int>()) {
            int samples = value.Get<int>();
            renderer_.set_target_spp(samples);
        }
    } else if (key == TfToken("kanahebi:global:depth")) {
        if (value.IsHolding<long>()) {
            long depth = value.Get<long>();
            renderer_.set_depth(depth);
        } else if (value.IsHolding<int>()) {
            int depth = value.Get<int>();
            renderer_.set_depth(depth);
        }
    } else if (key == TfToken("kanahebi:global:blockingmode")) {
        bool blocking = value.Get<bool>();
        renderer_.set_blocking_mode(blocking);
    } else if (key == TfToken("kanahebi:global:filmtransparent")) {
        bool film_transparent = value.Get<bool>();
        renderer_.set_film_transparent(film_transparent);
    } else if (key == TfToken("kanahebi:global:exposure")) {
        if (value.IsHolding<float>()) {
            float exposure = value.Get<float>();
            renderer_.set_exposure(exposure);
        } else if (value.IsHolding<double>()) {
            double exposure = value.Get<double>();
            renderer_.set_exposure(static_cast<float>(exposure));
        }
    } else if (key == TfToken("kanahebi:global:flipy")) {
        bool flip_y = value.Get<bool>();
        renderer_.set_flip_y(flip_y);
    }
}

VtDictionary HdKanahebiRenderDelegate::GetRenderStats() const {
    VtDictionary stats;

    stats["percent_complete"] = renderer_.percent_complete();
    stats["render_stage_label"] = renderer_.is_converged() ? "Converged" : "Rendering";

    return stats;
}

HdInstancer* HdKanahebiRenderDelegate::CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) {
    // Not implemented
    TF_CODING_ERROR("CreateInstancer is not implemented.");
    return nullptr;
}

void HdKanahebiRenderDelegate::DestroyInstancer(HdInstancer* instancer) {
    // Not implemented
    TF_CODING_ERROR("DestroyInstancer is not implemented.");
}

HdRprim* HdKanahebiRenderDelegate::CreateRprim(TfToken const& type_id, SdfPath const& rprim_id) {
    TF_CODING_ERROR("CreateRprim is not implemented.");
    return nullptr;
}

void HdKanahebiRenderDelegate::DestroyRprim(HdRprim* rprim) {
    // Not implemented
    TF_CODING_ERROR("DestroyRprim is not implemented.");
}

HdSprim* HdKanahebiRenderDelegate::CreateSprim(TfToken const& type_id, SdfPath const& sprim_id) {
#ifndef IS_BLENDER_4_5
    // Not implemented
    TF_CODING_ERROR("CreateSprim is not implemented.");
    return nullptr;
#else
    if (type_id == HdPrimTypeTokens->camera) {
        scene_.set_main_camera_path(sprim_id);
        return new HdCamera(sprim_id);
    }
    TF_CODING_ERROR("CreateSprim: Unsupported type_id %s", type_id.GetText());
    return nullptr;
#endif
}

HdSprim* HdKanahebiRenderDelegate::CreateFallbackSprim(TfToken const& type_id) {
#ifndef IS_BLENDER_4_5
    // Not implemented
    TF_CODING_ERROR("CreateFallbackSprim is not implemented.");
    return nullptr;
#else
    if (type_id == HdPrimTypeTokens->camera) {
        return new HdCamera(SdfPath());
    }
    return nullptr;
#endif
}

void HdKanahebiRenderDelegate::DestroySprim(HdSprim* sprim) {
#ifndef IS_BLENDER_4_5
    // Not implemented
    TF_CODING_ERROR("DestroySprim is not implemented.");
#else
    delete sprim;
#endif
}

HdBprim* HdKanahebiRenderDelegate::CreateBprim(TfToken const& type_id, SdfPath const& bprim_id) {
    if (type_id == HdPrimTypeTokens->renderBuffer) {
        return new HdKanahebiRenderBuffer(bprim_id, *hgi_);
    }
    TF_CODING_ERROR("CreateBprim: Unsupported type_id %s", type_id.GetText());
    return nullptr;
}

HdBprim* HdKanahebiRenderDelegate::CreateFallbackBprim(TfToken const& type_id) {
    if (type_id == HdPrimTypeTokens->renderBuffer) {
        return new HdKanahebiRenderBuffer(SdfPath(), *hgi_);
    }
    TF_CODING_ERROR("CreateFallbackBprim is not implemented.");
    return nullptr;
}

void HdKanahebiRenderDelegate::DestroyBprim(HdBprim* bprim) {
    delete bprim;
}

void HdKanahebiRenderDelegate::CommitResources(HdChangeTracker* tracker) {
    // Do nothing
}

HdAovDescriptor HdKanahebiRenderDelegate::GetDefaultAovDescriptor(TfToken const& aovName) const {
    HdAovDescriptor desc;
    if (aovName == HdAovTokens->color) {
#ifndef IS_BLENDER_4_5
        desc.format = HdFormatUNorm8Vec4;
        desc.clearValue = VtValue(GfVec4f(0.0f, 0.0f, 0.0f, 1.0f));
#else
        // BlenderではFloat32Vec4を要求する
        desc.format = HdFormatFloat32Vec4;
        desc.clearValue = VtValue(GfVec4f(0.0f, 0.0f, 0.0f, 1.0f));
#endif
    } else if (aovName == HdAovTokens->depth) {
        desc.format = HdFormatFloat32;
        desc.clearValue = VtValue(1.0f);
    }
    return desc;
}

HdRenderPassSharedPtr HdKanahebiRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                                                 HdRprimCollection const& collection) {
    return HdRenderPassSharedPtr(new HdKanahebiRenderPass(index, collection, renderer_));
}

void HdKanahebiRenderDelegate::SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& scene_index) {
    if (scene_index) {
        scene_index->AddObserver(HdSceneIndexObserverPtr(&scene_index_observer_));
    }
    scene_index_ = scene_index;
}

void HdKanahebiRenderDelegate::Update() {
    // Process scene index changes
    auto prim_changes = scene_index_observer_.consume_prim_changes();

    if (prim_changes.empty()) {
        return;
    }

    renderer_.begin_scene_update();

    cuda_context_.make_current();

    for (const auto& change : prim_changes) {
        std::visit(
                overloaded{[this](const HdSceneIndexObserver::AddedPrimEntries& entries) { prims_added(entries); },
                           [this](const HdSceneIndexObserver::RemovedPrimEntries& entries) { prims_removed(entries); },
                           [this](const HdSceneIndexObserver::DirtiedPrimEntries& entries) { prims_dirtied(entries); },
                           [this](const HdSceneIndexObserver::RenamedPrimEntries& entries) { prims_renamed(entries); }},
                change);
    }

    const ProgramGroups program_groups = renderer_.get_program_groups();
    std::unique_ptr<const SceneData> scene_data = scene_.build(program_groups);
    renderer_.set_scene_data(std::move(scene_data));

    renderer_.end_scene_update();
}

void HdKanahebiRenderDelegate::initialize() {
    std::cout << "HdKanahebiRenderDelegate initialized." << std::endl;

    // Initialize resource registry
    resource_registry_ = std::make_shared<HdResourceRegistry>();

    hgi_ = Hgi::CreatePlatformDefaultHgi();
}

void HdKanahebiRenderDelegate::prims_added(const HdSceneIndexObserver::AddedPrimEntries& entries) {
    for (const auto& entry : entries) {
        // std::cout << "Prim added: " << entry.primPath << " of type " << entry.primType << std::endl;
        // Additional logic for handling added prims can be added here

        scene_.add_prim(entry.primPath, scene_index_);
    }
}

void HdKanahebiRenderDelegate::prims_removed(const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    for (const auto& entry : entries) {
        // std::cout << "Prim removed: " << entry.primPath << std::endl;
        // Additional logic for handling removed prims can be added here

        scene_.delete_prim(entry.primPath, scene_index_);
    }
}

void HdKanahebiRenderDelegate::prims_dirtied(const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    for (const auto& entry : entries) {
        // std::cout << "Prim dirtied: " << entry.primPath << std::endl;
        // for (const auto& dirtyLocator : entry.dirtyLocators) {
        //     std::cout << "  Dirty locator: " << dirtyLocator << std::endl;
        // }
        // Additional logic for handling dirtied prims can be added here

        scene_.update_prim(entry.primPath, scene_index_);
    }
}

void HdKanahebiRenderDelegate::prims_renamed(const HdSceneIndexObserver::RenamedPrimEntries& entries) {
    for (const auto& entry : entries) {
        // std::cout << "Prim renamed from " << entry.oldPrimPath << " to " << entry.newPrimPath << std::endl;
        // Additional logic for handling renamed prims can be added here

        scene_.delete_prim(entry.oldPrimPath, scene_index_);
        scene_.add_prim(entry.newPrimPath, scene_index_);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
