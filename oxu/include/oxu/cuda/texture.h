#pragma once

#include <cuda.h>
#include <stdexcept>
#include "oxu/utils/error_check.h"

namespace oxu::cuda {

    using namespace utils;

    /// @brief CUDA Driver API 版の RAII Texture 管理クラス
    class Texture {
    public:
        Texture() = default;

        /// @brief CUDA Array + TextureObject を作成
        /// @param host_data RGBA8 のピクセルデータ
        /// @param width 幅
        /// @param height 高さ
        /// @param channels チャンネル数 (1, 2, 3, 4 のみ)
        /// @param address_mode アドレスモード (CU_TR_ADDRESS_MODE_WRAP など)
        /// @param filter_mode フィルタモード (CU_TR_FILTER_MODE_LINEAR / POINT)
        Texture(const void* host_data,
                int width,
                int height,
                int channels = 4,
                CUaddress_mode address_mode = CU_TR_ADDRESS_MODE_WRAP,
                CUfilter_mode filter_mode = CU_TR_FILTER_MODE_LINEAR)
            : width_(width), height_(height), channels_(channels) {
            if (!host_data || width <= 0 || height <= 0)
                throw std::runtime_error("invalid texture data");

            // チャンネルフォーマットを設定
            CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT8;
            unsigned int numChannels = 0;
            switch (channels) {
                case 1:
                    numChannels = 1;
                    break;
                case 2:
                    numChannels = 2;
                    break;
                case 3:
                case 4:
                    numChannels = 4;
                    break;  // CUDAはuchar3非対応
                default:
                    throw std::runtime_error("unsupported channel count");
            }

            CUDA_ARRAY3D_DESCRIPTOR arrayDesc{};
            arrayDesc.Width = width_;
            arrayDesc.Height = height_;
            arrayDesc.Depth = 0;
            arrayDesc.Format = format;
            arrayDesc.NumChannels = numChannels;
            arrayDesc.Flags = 0;

            // CUDA Array作成
            cu_check(cuArray3DCreate(&cu_array_, &arrayDesc));

            // ホストデータをコピー
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = host_data;
            copy.srcPitch = static_cast<size_t>(width_) * channels_ * sizeof(unsigned char);
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = cu_array_;
            copy.WidthInBytes = copy.srcPitch;
            copy.Height = height_;
            cu_check(cuMemcpy2D(&copy));

            // リソース記述子を設定
            CUDA_RESOURCE_DESC resDesc{};
            resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            resDesc.res.array.hArray = cu_array_;

            // テクスチャ記述子を設定
            CUDA_TEXTURE_DESC texDesc{};
            texDesc.addressMode[0] = address_mode;
            texDesc.addressMode[1] = address_mode;
            texDesc.filterMode = filter_mode;
            texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
            texDesc.maxAnisotropy = 1;
            texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            texDesc.mipmapLevelBias = 0.0f;
            texDesc.minMipmapLevelClamp = 0.0f;
            texDesc.maxMipmapLevelClamp = 0.0f;
            texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] = texDesc.borderColor[3] = 0.0f;

            // Texture Object作成
            cu_check(cuTexObjectCreate(&tex_obj_, &resDesc, &texDesc, nullptr));
        }

        // コピー禁止
        Texture(const Texture&) = delete;
        Texture& operator=(const Texture&) = delete;

        // ムーブ対応
        Texture(Texture&& other) noexcept
            : cu_array_(other.cu_array_),
              tex_obj_(other.tex_obj_),
              width_(other.width_),
              height_(other.height_),
              channels_(other.channels_) {
            other.cu_array_ = nullptr;
            other.tex_obj_ = 0;
        }

        Texture& operator=(Texture&& other) noexcept {
            if (this != &other) {
                cleanup();
                cu_array_ = other.cu_array_;
                tex_obj_ = other.tex_obj_;
                width_ = other.width_;
                height_ = other.height_;
                channels_ = other.channels_;
                other.cu_array_ = nullptr;
                other.tex_obj_ = 0;
            }
            return *this;
        }

        ~Texture() { cleanup(); }

        [[nodiscard]] CUtexObject handle() const noexcept { return tex_obj_; }
        [[nodiscard]] bool valid() const noexcept { return tex_obj_ != 0; }
        [[nodiscard]] int width() const noexcept { return width_; }
        [[nodiscard]] int height() const noexcept { return height_; }
        [[nodiscard]] int channels() const noexcept { return channels_; }

    private:
        void cleanup() {
            if (tex_obj_) {
                cuTexObjectDestroy(tex_obj_);
                tex_obj_ = 0;
            }
            if (cu_array_) {
                cuArrayDestroy(cu_array_);
                cu_array_ = nullptr;
            }
        }

        CUarray cu_array_{nullptr};
        CUtexObject tex_obj_{0};
        int width_{0};
        int height_{0};
        int channels_{0};
    };

}  // namespace oxu::cuda
