#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <optix.h>

#include "oxu/utils/error_check.h"

namespace oxu::optix {

    namespace fs = std::filesystem;

    using namespace utils;

    class Module {
    public:
        explicit Module(const OptixDeviceContext ctx, const OptixPipelineCompileOptions& pcomp, const fs::path ptx_path)
            : ctx_(ctx) {
            std::ifstream file(ptx_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open PTX file");
            }

            std::ostringstream buffer;
            buffer << file.rdbuf();
            const std::string ptx = buffer.str();

            OptixModuleCompileOptions mopt = {};
            mopt.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            mopt.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

            optix_check(optixModuleCreate(ctx, &mopt, &pcomp, ptx.c_str(), ptx.size(),
                                          /*logString*/ nullptr, /*logStringSize*/ nullptr, &module_));
        }

        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;

        Module(Module&& other) noexcept : ctx_(other.ctx_), module_(other.module_) { other.module_ = nullptr; }
        Module& operator=(Module&& other) noexcept {
            if (this != &other) {
                cleanup();
                ctx_ = other.ctx_;
                module_ = other.module_;
                other.module_ = nullptr;
            }
            return *this;
        }

        ~Module() { cleanup(); }

        // Conversion operator to OptixModule
        operator OptixModule() const { return module_; }

    private:
        void cleanup() {
            if (module_) {
                optix_check(optixModuleDestroy(module_));
                module_ = nullptr;
            }
        }

        OptixDeviceContext ctx_{nullptr};
        OptixModule module_{nullptr};
    };

}  // namespace oxu::optix
