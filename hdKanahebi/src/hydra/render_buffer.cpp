#include "hydra/render_buffer.h"

PXR_NAMESPACE_OPEN_SCOPE

HdKanahebiRenderBuffer::HdKanahebiRenderBuffer(SdfPath const& id, Hgi& hgi)
    : HdRenderBuffer(id),
      id_(id),
      dimensions_(0, 0, 0),
      format_(HdFormatInvalid),
      multi_sampled_(false),
      is_converged_(false),
      is_mapped_(false),
      is_dirty_texture_(false),
      hgi_(hgi) {}

bool HdKanahebiRenderBuffer::Allocate(GfVec3i const& dimensions, HdFormat format, bool multi_sampled) {
    if (dimensions[2] != 1) {
        TF_WARN("RenderBuffer allocated with dims <%d, %d, %d> and format %s; depth must be 1.", dimensions[0],
                dimensions[1], dimensions[2], TfEnum::GetName(format).c_str());
        return false;
    }

    if (dimensions == dimensions_ && format == format_) {
        // 変更なし
        return true;
    }

    dimensions_ = dimensions;
    format_ = format;
    multi_sampled_ = multi_sampled;  // マルチサンプルは関係ないので受け取ったものを保持して返すだけ

    is_converged_ = false;

    is_dirty_texture_.store(true);

    // ホストのバッファを確保
    size_t pixel_size = HdDataSizeOfFormat(format_);
    size_t total_size = dimensions_[0] * dimensions_[1] * dimensions_[2] * pixel_size;
    mapped_buffer_.resize(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        mapped_buffer_[i] = 0;
    }

    return true;
}

unsigned int HdKanahebiRenderBuffer::GetWidth() const {
    return dimensions_[0];
}

unsigned int HdKanahebiRenderBuffer::GetHeight() const {
    return dimensions_[1];
}

unsigned int HdKanahebiRenderBuffer::GetDepth() const {
    return dimensions_[2];
}

HdFormat HdKanahebiRenderBuffer::GetFormat() const {
    return format_;
}

bool HdKanahebiRenderBuffer::IsMultiSampled() const {
    return multi_sampled_;
}

void* HdKanahebiRenderBuffer::Map() {
    is_mapped_ = true;

    std::vector<uint8_t> data(buffer_.size_bytes());
    buffer_.download(std::span<uint8_t>(data.data(), data.size()));
    std::memcpy(mapped_buffer_.data(), data.data(), data.size());

    return mapped_buffer_.data();
}

void HdKanahebiRenderBuffer::Unmap() {
    is_mapped_ = false;
}

VtValue HdKanahebiRenderBuffer::GetResource(bool multiSampled) const {
    if (texture_) {
        return VtValue(texture_);
    } else {
        return VtValue();
    }
}

void HdKanahebiRenderBuffer::update_data(const oxu::cuda::Buffer& buffer, const bool converged) {
    create_texture_if_needed();

    if (hgi_.GetAPIName() == "OpenGL" && texture_ && is_backend_ready()) {
        if (!graphics_resource_.valid()) {
            GLuint texture_id = static_cast<GLuint>(texture_->GetRawResource());
            graphics_resource_ = oxu::cuda::GraphicsResource(texture_id);
        }
        graphics_resource_.upload_from_buffer(buffer, dimensions_[0], dimensions_[1], HdDataSizeOfFormat(format_));
    }

    if (buffer_.size_bytes() != buffer.size_bytes()) {
        buffer_ = oxu::cuda::Buffer(buffer.size_bytes());
    }
    buffer_.copy_from(buffer);

    is_converged_ = converged;
}

void HdKanahebiRenderBuffer::_Deallocate() {
    // Clear host buffer
    mapped_buffer_.clear();

    // Release Hgi texture
    if (texture_ && is_backend_ready()) {
        hgi_.DestroyTexture(&texture_);
    }
}

HgiFormat HdKanahebiRenderBuffer::convert_format(HdFormat hd_format) const {
    static const std::map<HdFormat, HgiFormat> FORMAT_DESC = {
            {HdFormatUNorm8, HgiFormatUNorm8},
            {HdFormatUNorm8Vec2, HgiFormatUNorm8Vec2},
            {HdFormatUNorm8Vec3, HgiFormatInvalid},  // Unsupported by HgiFormat
            {HdFormatUNorm8Vec4, HgiFormatUNorm8Vec4},

            {HdFormatSNorm8, HgiFormatSNorm8},
            {HdFormatSNorm8Vec2, HgiFormatSNorm8Vec2},
            {HdFormatSNorm8Vec3, HgiFormatInvalid},  // Unsupported by HgiFormat
            {HdFormatSNorm8Vec4, HgiFormatSNorm8Vec4},

            {HdFormatFloat16, HgiFormatFloat16},
            {HdFormatFloat16Vec2, HgiFormatFloat16Vec2},
            {HdFormatFloat16Vec3, HgiFormatFloat16Vec3},
            {HdFormatFloat16Vec4, HgiFormatFloat16Vec4},

            {HdFormatFloat32, HgiFormatFloat32},
            {HdFormatFloat32Vec2, HgiFormatFloat32Vec2},
            {HdFormatFloat32Vec3, HgiFormatFloat32Vec3},
            {HdFormatFloat32Vec4, HgiFormatFloat32Vec4},

            {HdFormatInt16, HgiFormatInt16},
            {HdFormatInt16Vec2, HgiFormatInt16Vec2},
            {HdFormatInt16Vec3, HgiFormatInt16Vec3},
            {HdFormatInt16Vec4, HgiFormatInt16Vec4},

            {HdFormatUInt16, HgiFormatUInt16},
            {HdFormatUInt16Vec2, HgiFormatUInt16Vec2},
            {HdFormatUInt16Vec3, HgiFormatUInt16Vec3},
            {HdFormatUInt16Vec4, HgiFormatUInt16Vec4},

            {HdFormatInt32, HgiFormatInt32},
            {HdFormatInt32Vec2, HgiFormatInt32Vec2},
            {HdFormatInt32Vec3, HgiFormatInt32Vec3},
            {HdFormatInt32Vec4, HgiFormatInt32Vec4},

            {HdFormatFloat32UInt8, HgiFormatFloat32UInt8},
    };

    auto it = FORMAT_DESC.find(hd_format);
    if (it != FORMAT_DESC.end()) {
        return it->second;
    } else {
        TF_CODING_ERROR("Unsupported HdFormat: %d", hd_format);
        return HgiFormatInvalid;
    }
}

void HdKanahebiRenderBuffer::create_texture_if_needed() {
    // テクスチャがダーティーな場合は再作成
    if (is_dirty_texture_.load() && is_backend_ready()) {
        // 既存のHgiテクスチャがあれば破棄
        if (texture_) {
            hgi_.DestroyTexture(&texture_);
        }

        size_t pixel_size = HdDataSizeOfFormat(format_);
        size_t total_size = dimensions_[0] * dimensions_[1] * dimensions_[2] * pixel_size;

        // Hgiのテクスチャを作成
        HgiTextureDesc tex_desc;
        tex_desc.format = convert_format(format_);
        tex_desc.dimensions = dimensions_;
        tex_desc.usage =
                HgiTextureUsageBitsColorTarget | HgiTextureUsageBitsShaderRead | HgiTextureUsageBitsShaderWrite;
        tex_desc.pixelsByteSize = total_size;
        tex_desc.initialData = nullptr;

        texture_ = hgi_.CreateTexture(tex_desc);

        graphics_resource_.invalidate();

        is_dirty_texture_.store(false);
    }
}

bool HdKanahebiRenderBuffer::is_backend_ready() const {
    if (hgi_.GetAPIName() == "OpenGL") {
        // OpenGLバックエンドの場合、コンテキストが有効か確認する
        // Houdini 21.0.512で、OpenGLのコンテキスト作成がHydraデリゲートの初期化よりあとに来る場合があり
        // またOpenGLコンテキストの破棄がHydraデリゲートの破棄より前に来る場合があるため
        GlfGLContextSharedPtr gl_context = GlfGLContext::GetCurrentGLContext();
        if (gl_context && gl_context->IsValid()) {
            return true;
        } else {
            return false;
        }
    } else {
        // 他のバックエンドの場合は常に準備完了とみなす
        return true;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
