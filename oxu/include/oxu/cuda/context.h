#pragma once

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "oxu/utils/error_check.h"

namespace oxu::cuda {

    using namespace utils;

    class Context {
    public:
        Context(const int device_id = 0) {
            cu_check(cuInit(0));
            cu_check(cuDeviceGet(&device_, device_id));
            cu_check(cuCtxCreate(&context_, nullptr, 0, device_));
        }

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        Context(Context&& other) noexcept : device_(other.device_), context_(other.context_) {
            other.device_ = 0;
            other.context_ = nullptr;
        }
        Context& operator=(Context&& other) noexcept {
            if (this != &other) {
                cleanup();
                device_ = other.device_;
                context_ = other.context_;
                other.device_ = 0;
                other.context_ = nullptr;
            }
            return *this;
        }

        ~Context() { cleanup(); }

        void make_current() const { cu_check(cuCtxSetCurrent(context_)); }

        CUdevice device() const { return device_; }

        // Conversion operator to CUcontext
        operator CUcontext() const { return context_; }

    private:
        void cleanup() {
            if (context_) {
                cu_check(cuCtxDestroy(context_));
                context_ = nullptr;
            }
            device_ = 0;
        }

        CUdevice device_{0};
        CUcontext context_{nullptr};
    };

}  // namespace oxu::cuda
