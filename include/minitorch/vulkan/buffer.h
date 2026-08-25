#pragma once

#include "minitorch/vulkan/context.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minitorch {

class VulkanBuffer {
public:
    VulkanBuffer();
    
    VulkanBuffer(
        VulkanContext& context,
        std::size_t size
    );

    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    bool valid() const;

    std::size_t size() const;

    VkBuffer handle() const;

    void upload(
        const void* data,
        std::size_t size
    );

    void download(
        void* data,
        std::size_t size
    ) const;

    void upload_f32(
        const std::vector<float>& data
    );

    std::vector<float> download_f32(
        std::size_t count
    ) const;

private:
    VulkanContext* context_;

    VkBuffer buffer_;
    VkDeviceMemory memory_;

    std::size_t size_;

    uint32_t memory_type_index_;

    void release();

    uint32_t find_memory_type(
        uint32_t type_filter,
        VkMemoryPropertyFlags properties
    ) const;
};

} // namespace minitorch
