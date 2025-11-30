#pragma once

#include <iostream>
#include <string>

#include <pxr/base/tf/staticTokens.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/pxr.h>

#include <oxu/cuda/context.h>
#include <oxu/optix/context.h>

#include "hydra/render_buffer.h"
#include "hydra/render_pass.h"
#include "hydra/scene_index_observer.h"
#include "overloaded.h"
#include "renderer/renderer.h"
#include "renderer/scene.h"

PXR_NAMESPACE_OPEN_SCOPE

/// @brief KanahebiレンダラーのHdRenderDelegateの実装クラス
class HdKanahebiRenderDelegate final : public HdRenderDelegate {
public:
    HdKanahebiRenderDelegate();
    HdKanahebiRenderDelegate(HdRenderSettingsMap const& settings_map);
    ~HdKanahebiRenderDelegate() override;

    // This class does not support copying.
    HdKanahebiRenderDelegate(HdKanahebiRenderDelegate const&) = delete;
    HdKanahebiRenderDelegate& operator=(HdKanahebiRenderDelegate const&) = delete;

    const TfTokenVector& GetSupportedRprimTypes() const override;
    const TfTokenVector& GetSupportedSprimTypes() const override;
    const TfTokenVector& GetSupportedBprimTypes() const override;

    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    void SetRenderSetting(TfToken const& key, VtValue const& value) override;

    VtDictionary GetRenderStats() const override;

    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) override;
    void DestroyInstancer(HdInstancer* instancer) override;

    HdRprim* CreateRprim(TfToken const& type_id, SdfPath const& rprim_id) override;
    void DestroyRprim(HdRprim* rprim) override;

    HdSprim* CreateSprim(TfToken const& type_id, SdfPath const& sprim_id) override;
    HdSprim* CreateFallbackSprim(TfToken const& type_id) override;
    void DestroySprim(HdSprim* sprim) override;

    HdBprim* CreateBprim(TfToken const& type_id, SdfPath const& bprim_id) override;
    HdBprim* CreateFallbackBprim(TfToken const& type_id) override;
    void DestroyBprim(HdBprim* bprim) override;

    void CommitResources(HdChangeTracker* tracker) override;

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const& aovName) const override;

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index, HdRprimCollection const& collection) override;

    void SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& scene_index) override;

    void Update() override;

private:
    /// @brief RenderDelegateの初期化を行います。
    void initialize();

    /// @brief 新しいプリミティブの追加を処理します。
    /// @param entries 追加されたプリミティブのエントリ
    void prims_added(const HdSceneIndexObserver::AddedPrimEntries& entries);

    /// @brief プリミティブの削除を処理します。
    /// @param entries 削除されたプリミティブのエントリ
    void prims_removed(const HdSceneIndexObserver::RemovedPrimEntries& entries);

    /// @brief プリミティブの変更を処理します。
    /// @param entries 変更されたプリミティブのエントリ
    void prims_dirtied(const HdSceneIndexObserver::DirtiedPrimEntries& entries);

    /// @brief プリミティブの名前変更を処理します。
    /// @param entries 名前が変更されたプリミティブのエントリ
    void prims_renamed(const HdSceneIndexObserver::RenamedPrimEntries& entries);

    HgiUniquePtr hgi_;
    HdResourceRegistrySharedPtr resource_registry_;
    HdSceneIndexBaseRefPtr scene_index_;
    HdKanahebiSceneIndexObserver scene_index_observer_;

    oxu::cuda::Context cuda_context_;
    oxu::optix::Context optix_context_;
    Renderer renderer_;
    Scene scene_;
};

PXR_NAMESPACE_CLOSE_SCOPE
