#pragma once

#include <span>
#include <stdexcept>
#include <type_traits>

#include <cuda.h>

#include "oxu/utils/error_check.h"

namespace oxu::cuda {

    using namespace utils;

    class Buffer {
    public:
        explicit Buffer(const size_t size_bytes = 0) : size_bytes_(size_bytes) {
            if (size_bytes_ > 0)
                cu_check(cuMemAlloc(&device_ptr_, size_bytes_));
        }

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept : device_ptr_(other.device_ptr_), size_bytes_(other.size_bytes_) {
            other.device_ptr_ = 0;
            other.size_bytes_ = 0;
        }

        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                cleanup();
                device_ptr_ = other.device_ptr_;
                size_bytes_ = other.size_bytes_;
                other.device_ptr_ = 0;
                other.size_bytes_ = 0;
            }
            return *this;
        }

        ~Buffer() { cleanup(); }

        [[nodiscard]] CUdeviceptr device_ptr() noexcept { return device_ptr_; }
        [[nodiscard]] const CUdeviceptr device_ptr() const noexcept { return device_ptr_; }
        [[nodiscard]] size_t size_bytes() const noexcept { return size_bytes_; }
        [[nodiscard]] bool valid() const noexcept { return device_ptr_ != 0; }

        /// @brief Copies data from the host to the device.
        /// @tparam T
        /// @param data
        template <typename T>
            requires std::is_trivially_copyable_v<T>
        void upload(std::span<const T> data) const {
            const size_t bytes = data.size_bytes();
            if (bytes == 0)
                return;
            if (bytes > size_bytes_)
                throw std::runtime_error("upload out of range");
            cu_check(cuMemcpyHtoD(device_ptr_, data.data(), bytes));
        }

        /// @brief Copies data from the device to the host.
        /// @tparam T
        /// @param data
        template <typename T>
            requires std::is_trivially_copyable_v<T>
        void download(std::span<T> data) const {
            const size_t bytes = data.size_bytes();
            if (bytes == 0)
                return;
            if (bytes > size_bytes_)
                throw std::runtime_error("download out of range");
            cu_check(cuMemcpyDtoH(data.data(), device_ptr_, bytes));
        }

        /// @brief Copies data from another device buffer to this buffer.
        /// @param src
        /// @param bytes
        /// @param dstOffset
        /// @param srcOffset
        void copy_from(const Buffer& src, size_t bytes, size_t dst_offset = 0, size_t src_offset = 0) const {
            if (bytes == 0)
                return;
            if (dst_offset + bytes > size_bytes_ || src_offset + bytes > src.size_bytes_)
                throw std::runtime_error("device-to-device copy out of range");
            cu_check(cuMemcpyDtoD(device_ptr_ + dst_offset, src.device_ptr_ + src_offset, bytes));
        }

        /// @brief Copies the entire contents of the source buffer to this buffer.
        /// @param src
        void copy_from(const Buffer& src) const {
            if (size_bytes_ != src.size_bytes_)
                throw std::runtime_error("buffer size mismatch");
            if (size_bytes_ == 0)
                return;
            cu_check(cuMemcpyDtoD(device_ptr_, src.device_ptr_, size_bytes_));
        }

        /// @brief Sets the contents of the buffer to a specific value.
        /// @param value
        void memset(uint8_t value = 0) const {
            if (!valid())
                return;
            cu_check(cuMemsetD8(device_ptr_, value, size_bytes_));
        }

    private:
        void cleanup() {
            if (device_ptr_) {
                cu_check(cuMemFree(device_ptr_));
                device_ptr_ = 0;
                size_bytes_ = 0;
            }
        }

        CUdeviceptr device_ptr_{0};
        size_t size_bytes_{0};
    };

}  // namespace oxu::cuda
