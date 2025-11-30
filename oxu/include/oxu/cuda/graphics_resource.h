#pragma once

#include <iostream>

#include <cuda.h>
#include <cudaGL.h>

#include "oxu/cuda/buffer.h"
#include "oxu/utils/error_check.h"

namespace oxu::cuda {

    using namespace utils;

    class GraphicsResource {
    public:
        explicit GraphicsResource() = default;
        explicit GraphicsResource(GLuint texture_id) {
            cu_check(cuGraphicsGLRegisterImage(&resource_, texture_id, GL_TEXTURE_2D,
                                               CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
        }

        GraphicsResource(const GraphicsResource&) = delete;
        GraphicsResource& operator=(const GraphicsResource&) = delete;

        GraphicsResource(GraphicsResource&& other) noexcept : resource_(other.resource_) { other.resource_ = nullptr; }

        GraphicsResource& operator=(GraphicsResource&& other) noexcept {
            if (this != &other) {
                cleanup();
                resource_ = other.resource_;
                other.resource_ = nullptr;
            }
            return *this;
        }

        ~GraphicsResource() { cleanup(); }

        [[nodiscard]] CUgraphicsResource get() const noexcept { return resource_; }
        [[nodiscard]] bool valid() const noexcept { return resource_ != nullptr; }

        void upload_from_buffer(const Buffer& buffer,
                                unsigned int width,
                                unsigned int height,
                                unsigned int bytes_per_pixel,
                                unsigned int array_index = 0,
                                unsigned int mip_level = 0) {
            if (!buffer.valid())
                throw std::runtime_error("Buffer is not valid");
            if (!valid())
                throw std::runtime_error("GraphicsResource is not valid");

            map();

            CUarray array = get_mapped_array(array_index, mip_level);

            CUDA_MEMCPY2D copyParam = {};
            copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParam.srcDevice = buffer.device_ptr();
            copyParam.srcPitch = width * bytes_per_pixel;

            copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copyParam.dstArray = array;

            copyParam.WidthInBytes = width * bytes_per_pixel;
            copyParam.Height = height;

            cu_check(cuMemcpy2D(&copyParam));

            unmap();
        }

        void invalidate() { cleanup(); }

    private:
        void cleanup() {
            if (resource_) {
                // プラットフォームによってはcleanup時にOpenGLのコンテキストがすでに存在せずエラーになることがあるため、
                // 安全のためコンテキストが有効か確認する
                const GLubyte* version = glGetString(GL_VERSION);
                GLenum error = glGetError();
                if (version != NULL && error == GL_NO_ERROR) {
                    cu_check(cuGraphicsUnregisterResource(resource_));
                }

                resource_ = nullptr;
            }
        }

        void map() { cu_check(cuGraphicsMapResources(1, &resource_, 0)); }

        void unmap() { cu_check(cuGraphicsUnmapResources(1, &resource_, 0)); }

        [[nodiscard]] CUarray get_mapped_array(unsigned int array_index = 0, unsigned int mip_level = 0) const {
            CUarray array = nullptr;
            cu_check(cuGraphicsSubResourceGetMappedArray(&array, resource_, array_index, mip_level));
            return array;
        }

        CUgraphicsResource resource_{nullptr};
    };

}  // namespace oxu::cuda
