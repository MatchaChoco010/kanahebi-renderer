#pragma once

#include <iostream>
#include <string>

#include <optix.h>
#include <optix_stubs.h>

#include "oxu/utils/error_check.h"

namespace oxu::optix {

    using namespace utils;

    class Context {
    public:
        explicit Context(const CUcontext cuCtx) {
            optix_check(optixInit());

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = [](unsigned int, const char* tag, const char* msg, void*) {
                std::cerr << "[OptiX]" << (tag ? std::string(" [") + tag + "] " : " ") << (msg ? msg : "") << "\n";
            };
            options.logCallbackLevel = 3;
            optix_check(optixDeviceContextCreate(cuCtx, &options, &context_));
        }

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        Context(Context&& other) noexcept : context_(other.context_) { other.context_ = nullptr; }
        Context& operator=(Context&& other) noexcept {
            if (this != &other) {
                cleanup();
                context_ = other.context_;
                other.context_ = nullptr;
            }
            return *this;
        }

        ~Context() { cleanup(); }

        // Conversion operator to OptixDeviceContext
        operator OptixDeviceContext() const { return context_; }

    private:
        void cleanup() {
            if (context_) {
                optix_check(optixDeviceContextDestroy(context_));
                context_ = nullptr;
            }
        }

        OptixDeviceContext context_{nullptr};
    };

}  // namespace oxu::optix
