#pragma once

#include "minitorch/vulkan/context.h"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace minitorch {

class VulkanCompute {
public:
    explicit VulkanCompute(VulkanContext& context);

    ~VulkanCompute();

    VulkanCompute(const VulkanCompute&) = delete;
    VulkanCompute& operator=(const VulkanCompute&) = delete;

    VulkanCompute(VulkanCompute&& other) noexcept;
    VulkanCompute& operator=(VulkanCompute&& other) noexcept;

    bool valid() const;

    VkCommandPool command_pool() const;
    VkCommandBuffer command_buffer() const;
    VkFence fence() const;

    void begin();
    void end();
    void submit_and_wait();

private:
    VulkanContext* context_;

    VkCommandPool command_pool_;
    VkCommandBuffer command_buffer_;
    VkFence fence_;

    bool recording_;

    void create_command_pool();
    void allocate_command_buffer();
    void create_fence();

    void destroy();

    void check(
        VkResult result,
        const char* operation
    ) const;
};

} // namespace minitorch
