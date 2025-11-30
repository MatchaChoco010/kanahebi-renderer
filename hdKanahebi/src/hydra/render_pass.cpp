#include "hydra/render_pass.h"

PXR_NAMESPACE_OPEN_SCOPE

HdKanahebiRenderPass::HdKanahebiRenderPass(HdRenderIndex* index,
                                           HdRprimCollection const& collection,
                                           Renderer& renderer)
    : HdRenderPass(index, collection), renderer_(renderer) {}

HdKanahebiRenderPass::~HdKanahebiRenderPass() {}

void HdKanahebiRenderPass::_Execute(HdRenderPassStateSharedPtr const& render_pass_state,
                                    TfTokenVector const& render_tags) {
    renderer_.sync_render_buffers(render_pass_state->GetAovBindings());
    converged_ = renderer_.is_converged();
}

PXR_NAMESPACE_CLOSE_SCOPE
