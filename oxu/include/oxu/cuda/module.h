#pragma once

#include <filesystem>

#include <cuda.h>

#include "oxu/cuda/context.h"
#include "oxu/utils/error_check.h"

namespace oxu::cuda {

    namespace fs = std::filesystem;

    using namespace utils;

    class Module {
    public:
        explicit Module(const CUcontext context, const fs::path ptx_path) {
#ifdef _WIN32
            auto utf8_path = ptx_path.u8string();
            std::string path(utf8_path.begin(), utf8_path.end());
            cu_check(cuModuleLoad(&module_, path.c_str()));
#else
            cu_check(cuModuleLoad(&module_, ptx_path.c_str()));
#endif
        }

        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;

        Module(Module&& other) noexcept : module_(other.module_) { other.module_ = nullptr; }
        Module& operator=(Module&& other) noexcept {
            if (this != &other) {
                cleanup();
                module_ = other.module_;
                other.module_ = nullptr;
            }
            return *this;
        }

        ~Module() { cleanup(); }

        CUfunction get_function(const std::string& name) const {
            CUfunction func;
            cu_check(cuModuleGetFunction(&func, module_, name.c_str()));
            return func;
        }

        // Conversion operator to CUmodule
        operator CUmodule() const { return module_; }

    private:
        void cleanup() {
            if (module_) {
                cu_check(cuModuleUnload(module_));
                module_ = nullptr;
            }
        }

        CUmodule module_{nullptr};
    };

}  // namespace oxu::cuda
