#pragma once

#include <iostream>
#include <string>

#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/pxr.h>

#include "hydra/render_delegate.h"

PXR_NAMESPACE_OPEN_SCOPE

/// @brief KanahebiレンダラーのHdRendererPluginの実装クラス
class HdKanahebiRendererPlugin final : public HdRendererPlugin {
public:
    HdKanahebiRendererPlugin() = default;
    ~HdKanahebiRendererPlugin() override = default;

    // This class does not support copying.
    HdKanahebiRendererPlugin(HdKanahebiRendererPlugin const&) = delete;
    HdKanahebiRendererPlugin& operator=(HdKanahebiRendererPlugin const&) = delete;

    HdRenderDelegate* CreateRenderDelegate() override;

    HdRenderDelegate* CreateRenderDelegate(HdRenderSettingsMap const& settings_map) override;

    void DeleteRenderDelegate(HdRenderDelegate* render_delegate) override;

    bool IsSupported(bool gpu_enabled = true) const override;

#ifdef PXR_VERSION_v25_11
    bool IsSupported(HdRendererCreateArgs const& rendererCreateArgs,
                     std::string* reasonWhyNot = nullptr) const override;
#endif
};

PXR_NAMESPACE_CLOSE_SCOPE
