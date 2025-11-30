#pragma once

#include <span>

#include <optix.h>

#include "oxu/utils/error_check.h"

namespace oxu::optix {

    using namespace utils;

    class Pipeline {
    public:
        explicit Pipeline(const OptixDeviceContext ctx,
                          const OptixPipelineCompileOptions* pcomp,
                          const OptixPipelineLinkOptions* linkopt,
                          const std::span<OptixProgramGroup> program_groups,
                          uint32_t continuation_stack_size,
                          uint32_t max_traversable_graph_depth)
            : ctx_(ctx) {
            optix_check(optixPipelineCreate(ctx, pcomp, linkopt, program_groups.data(), program_groups.size(),
                                            /*logString*/ nullptr, /*logStringSize*/ nullptr, &pipeline_));

            // Set stack size
            uint32_t direct_callable_stack_size_from_traversal = 0;
            uint32_t direct_callable_stack_size_from_state = 0;
            optix_check(optixPipelineSetStackSize(pipeline_, direct_callable_stack_size_from_traversal,
                                                  direct_callable_stack_size_from_state, continuation_stack_size,
                                                  max_traversable_graph_depth));
        }

        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        Pipeline(Pipeline&& other) noexcept : ctx_(other.ctx_), pipeline_(other.pipeline_) {
            other.pipeline_ = nullptr;
        }
        Pipeline& operator=(Pipeline&& other) noexcept {
            if (this != &other) {
                cleanup();
                ctx_ = other.ctx_;
                pipeline_ = other.pipeline_;
                other.pipeline_ = nullptr;
            }

            return *this;
        }
        ~Pipeline() { cleanup(); }

        // Conversion operator to OptixPipeline
        operator OptixPipeline() const { return pipeline_; }

    private:
        void cleanup() {
            if (pipeline_) {
                optix_check(optixPipelineDestroy(pipeline_));
                pipeline_ = nullptr;
            }
        }

        OptixDeviceContext ctx_{nullptr};
        OptixPipeline pipeline_{nullptr};
    };

}  // namespace oxu::optix
