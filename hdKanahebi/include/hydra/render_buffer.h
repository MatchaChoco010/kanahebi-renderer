#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <pxr/base/gf/vec3i.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/glf/glContext.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hgi/blitCmds.h>
#include <pxr/imaging/hgi/blitCmdsOps.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/texture.h>
#include <pxr/pxr.h>

#ifdef _WIN32
#include <windows.h>

#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <oxu/cuda/buffer.h>
#include <oxu/cuda/graphics_resource.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdKanahebiRenderBuffer final : public HdRenderBuffer {
public:
    HdKanahebiRenderBuffer(SdfPath const& id, Hgi& hgi);

    bool Allocate(GfVec3i const& dimensions, HdFormat format, bool multiSampled) override;

    unsigned int GetWidth() const override;
    unsigned int GetHeight() const override;
    unsigned int GetDepth() const override;
    HdFormat GetFormat() const override;
    bool IsMultiSampled() const override;

    void* Map() override;
    void Unmap() override;
    bool IsMapped() const override { return is_mapped_; }

    void Resolve() override {};

    bool IsConverged() const override { return is_converged_; }

    VtValue GetResource(bool multiSampled) const override;

    void update_data(const oxu::cuda::Buffer& buffer, const bool converged);

private:
    void _Deallocate() override;

    HgiFormat convert_format(HdFormat hd_format) const;
    void create_texture_if_needed();
    bool is_backend_ready() const;

    SdfPath id_;

    GfVec3i dimensions_;
    HdFormat format_;
    bool multi_sampled_;
    bool is_converged_;
    bool is_mapped_;

    Hgi& hgi_;
    oxu::cuda::Buffer buffer_;
    std::vector<uint8_t> mapped_buffer_;
    std::atomic<bool> is_dirty_texture_;

    HgiTextureHandle texture_;
    oxu::cuda::GraphicsResource graphics_resource_;
};

PXR_NAMESPACE_CLOSE_SCOPE
