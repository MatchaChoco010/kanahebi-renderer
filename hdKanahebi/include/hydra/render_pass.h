#pragma once

#include <iostream>

#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/pxr.h>

#include "hydra/render_buffer.h"
#include "renderer/renderer.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdKanahebiRenderPass final : public HdRenderPass {
public:
    HdKanahebiRenderPass(HdRenderIndex* index, HdRprimCollection const& collection, Renderer& renderer);
    ~HdKanahebiRenderPass() override;

    bool IsConverged() const override { return converged_; };

private:
    void _Execute(HdRenderPassStateSharedPtr const& render_pass_state, TfTokenVector const& render_tags) override;

    bool converged_ = false;
    Renderer& renderer_;
};

PXR_NAMESPACE_CLOSE_SCOPE
