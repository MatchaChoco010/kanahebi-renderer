#pragma once

#include <optix.h>

#include "oxu/utils/error_check.h"

namespace oxu::optix {

    using namespace utils;

    class ProgramGroup {
    public:
        explicit ProgramGroup(const OptixDeviceContext ctx,
                              const OptixProgramGroupDesc& desc,
                              const OptixProgramGroupOptions& options)
            : ctx_(ctx) {
            optix_check(optixProgramGroupCreate(ctx, &desc, 1, &options,
                                                /*logString*/ nullptr, /*logStringSize*/ nullptr, &pg_));
        }

        ProgramGroup(const ProgramGroup&) = delete;
        ProgramGroup& operator=(const ProgramGroup&) = delete;

        ProgramGroup(ProgramGroup&& other) noexcept : ctx_(other.ctx_), pg_(other.pg_) { other.pg_ = nullptr; }
        ProgramGroup& operator=(ProgramGroup&& other) noexcept {
            if (this != &other) {
                cleanup();
                ctx_ = other.ctx_;
                pg_ = other.pg_;
                other.pg_ = nullptr;
            }
            return *this;
        }

        ~ProgramGroup() { cleanup(); }

        operator OptixProgramGroup() const { return pg_; }

    private:
        void cleanup() {
            if (pg_) {
                optix_check(optixProgramGroupDestroy(pg_));
                pg_ = nullptr;
            }
        }

        OptixDeviceContext ctx_{nullptr};
        OptixProgramGroup pg_{nullptr};
    };

}  // namespace oxu::optix
