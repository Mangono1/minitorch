#include "minitorch/vulkan/compute.h"

#include <stdexcept>
#include <utility>

namespace minitorch {

VulkanCompute::VulkanCompute(
    VulkanContext& context
)
    : context_(&context),
      command_pool_(VK_NULL_HANDLE),
      command_buffer_(VK_NULL_HANDLE),
      fence_(VK_NULL_HANDLE),
      recording_(false) {

    if (!context.available()) {
        throw std::runtime_error(
            "VulkanCompute requires an available Vulkan context"
        );
    }

    create_command_pool();
    allocate_command_buffer();
    create_fence();
}

VulkanCompute::~VulkanCompute() {
    destroy();
}

VulkanCompute::VulkanCompute(
    VulkanCompute&& other
) noexcept
    : context_(other.context_),
      command_pool_(other.command_pool_),
      command_buffer_(other.command_buffer_),
      fence_(other.fence_),
      recording_(other.recording_) {

    other.context_ = nullptr;
    other.command_pool_ = VK_NULL_HANDLE;
    other.command_buffer_ = VK_NULL_HANDLE;
    other.fence_ = VK_NULL_HANDLE;
    other.recording_ = false;
}

VulkanCompute& VulkanCompute::operator=(
    VulkanCompute&& other
) noexcept {

    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = other.context_;
    command_pool_ = other.command_pool_;
    command_buffer_ = other.command_buffer_;
    fence_ = other.fence_;
    recording_ = other.recording_;

    other.context_ = nullptr;
    other.command_pool_ = VK_NULL_HANDLE;
    other.command_buffer_ = VK_NULL_HANDLE;
    other.fence_ = VK_NULL_HANDLE;
    other.recording_ = false;

    return *this;
}

bool VulkanCompute::valid() const {
    return
        context_ != nullptr &&
        context_->available() &&
        command_pool_ != VK_NULL_HANDLE &&
        command_buffer_ != VK_NULL_HANDLE &&
        fence_ != VK_NULL_HANDLE;
}

VkCommandPool VulkanCompute::command_pool() const {
    return command_pool_;
}

VkCommandBuffer VulkanCompute::command_buffer() const {
    return command_buffer_;
}

VkFence VulkanCompute::fence() const {
    return fence_;
}

void VulkanCompute::create_command_pool() {

    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

    info.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    info.queueFamilyIndex =
        context_->compute_queue_family();

    check(
        vkCreateCommandPool(
            context_->device(),
            &info,
            nullptr,
            &command_pool_
        ),
        "vkCreateCommandPool"
    );
}

void VulkanCompute::allocate_command_buffer() {

    VkCommandBufferAllocateInfo info{};
    info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

    info.commandPool = command_pool_;

    info.level =
        VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    info.commandBufferCount = 1;

    check(
        vkAllocateCommandBuffers(
            context_->device(),
            &info,
            &command_buffer_
        ),
        "vkAllocateCommandBuffers"
    );
}

void VulkanCompute::create_fence() {

    VkFenceCreateInfo info{};
    info.sType =
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    info.flags = 0;

    check(
        vkCreateFence(
            context_->device(),
            &info,
            nullptr,
            &fence_
        ),
        "vkCreateFence"
    );
}

void VulkanCompute::begin() {

    if (!valid()) {
        throw std::runtime_error(
            "Invalid VulkanCompute"
        );
    }

    if (recording_) {
        throw std::runtime_error(
            "VulkanCompute command buffer is already recording"
        );
    }

    check(
        vkResetCommandBuffer(
            command_buffer_,
            0
        ),
        "vkResetCommandBuffer"
    );

    VkCommandBufferBeginInfo info{};
    info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    info.flags = 0;

    check(
        vkBeginCommandBuffer(
            command_buffer_,
            &info
        ),
        "vkBeginCommandBuffer"
    );

    recording_ = true;
}

void VulkanCompute::end() {

    if (!recording_) {
        throw std::runtime_error(
            "VulkanCompute command buffer is not recording"
        );
    }

    check(
        vkEndCommandBuffer(
            command_buffer_
        ),
        "vkEndCommandBuffer"
    );

    recording_ = false;
}

void VulkanCompute::submit_and_wait() {

    if (!valid()) {
        throw std::runtime_error(
            "Invalid VulkanCompute"
        );
    }

    if (recording_) {
        throw std::runtime_error(
            "Command buffer must be ended before submission"
        );
    }

    check(
        vkResetFences(
            context_->device(),
            1,
            &fence_
        ),
        "vkResetFences"
    );

    VkSubmitInfo submit{};
    submit.sType =
        VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;

    check(
        vkQueueSubmit(
            context_->compute_queue(),
            1,
            &submit,
            fence_
        ),
        "vkQueueSubmit"
    );

    check(
        vkWaitForFences(
            context_->device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX
        ),
        "vkWaitForFences"
    );
}

void VulkanCompute::destroy() {

    if (context_ == nullptr) {
        return;
    }

    VkDevice device = context_->device();

    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(
            device,
            fence_,
            nullptr
        );

        fence_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(
            device,
            command_pool_,
            nullptr
        );

        command_pool_ = VK_NULL_HANDLE;
        command_buffer_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    recording_ = false;
}

void VulkanCompute::check(
    VkResult result,
    const char* operation
) const {

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) +
            " failed with VkResult " +
            std::to_string(
                static_cast<int>(result)
            )
        );
    }
}

} // namespace minitorch
