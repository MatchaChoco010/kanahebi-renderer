#pragma once

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace oxu::utils {

    inline void cuda_check(cudaError_t result) {
        if (result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(result));
        }
    }

    inline void cu_check(CUresult result) {
        if (result != CUDA_SUCCESS) {
            const char* err = nullptr;
            cuGetErrorString(result, &err);
            throw std::runtime_error(err ? err : "CUDA driver error");
        }
    }

    inline void optix_check(OptixResult result) {
        if (result != OPTIX_SUCCESS) {
            throw std::runtime_error(std::string("OptiX error: ") + std::to_string(result));
        }
    }

}  // namespace oxu::utils
