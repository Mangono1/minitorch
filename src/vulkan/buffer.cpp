#include "minitorch/vulkan/buffer.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace minitorch {

VulkanBuffer::VulkanBuffer()
    : context_(nullptr),
      buffer_(VK_NULL_HANDLE),
      memory_(VK_NULL_HANDLE),
      size_(0),
      memory_type_index_(0) {
}

VulkanBuffer::VulkanBuffer(
    VulkanContext& context,
    std::size_t size
)
    : context_(&context),
      buffer_(VK_NULL_HANDLE),
      memory_(VK_NULL_HANDLE),
      size_(size),
      memory_type_index_(0) {

    if (size == 0) {
        throw std::invalid_argument(
            "VulkanBuffer size must be greater than zero"
        );
    }

    VkDevice device =
        context.device();

    if (device == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "VulkanBuffer requires a valid Vulkan device"
        );
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    buffer_info.size = size;

    buffer_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    buffer_info.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;

    VkResult result =
        vkCreateBuffer(
            device,
            &buffer_info,
            nullptr,
            &buffer_
        );

    if (result != VK_SUCCESS) {
        buffer_ = VK_NULL_HANDLE;

        throw std::runtime_error(
            "Failed to create Vulkan buffer"
        );
    }

    VkMemoryRequirements requirements{};

    vkGetBufferMemoryRequirements(
        device,
        buffer_,
        &requirements
    );

    memory_type_index_ =
        find_memory_type(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

    VkMemoryAllocateInfo allocate_info{};

    allocate_info.sType =
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    allocate_info.allocationSize =
        requirements.size;

    allocate_info.memoryTypeIndex =
        memory_type_index_;

    result =
        vkAllocateMemory(
            device,
            &allocate_info,
            nullptr,
            &memory_
        );

    if (result != VK_SUCCESS) {
        release();

        throw std::runtime_error(
            "Failed to allocate Vulkan buffer memory"
        );
    }

    result =
        vkBindBufferMemory(
            device,
            buffer_,
            memory_,
            0
        );

    if (result != VK_SUCCESS) {
        release();

        throw std::runtime_error(
            "Failed to bind Vulkan buffer memory"
        );
    }
}

VulkanBuffer::~VulkanBuffer() {
    release();
}

VulkanBuffer::VulkanBuffer(
    VulkanBuffer&& other
) noexcept
    : context_(other.context_),
      buffer_(other.buffer_),
      memory_(other.memory_),
      size_(other.size_),
      memory_type_index_(other.memory_type_index_) {

    other.context_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.memory_type_index_ = 0;
}

VulkanBuffer& VulkanBuffer::operator=(
    VulkanBuffer&& other
) noexcept {

    if (this == &other) {
        return *this;
    }

    release();

    context_ = other.context_;
    buffer_ = other.buffer_;
    memory_ = other.memory_;
    size_ = other.size_;
    memory_type_index_ =
        other.memory_type_index_;

    other.context_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.memory_type_index_ = 0;

    return *this;
}

bool VulkanBuffer::valid() const {
    return
        context_ != nullptr &&
        buffer_ != VK_NULL_HANDLE &&
        memory_ != VK_NULL_HANDLE;
}

std::size_t VulkanBuffer::size() const {
    return size_;
}

VkBuffer VulkanBuffer::handle() const {
    return buffer_;
}

void VulkanBuffer::upload(
    const void* data,
    std::size_t size
) {
    if (!valid()) {
        throw std::runtime_error(
            "Cannot upload to invalid Vulkan buffer"
        );
    }

    if (data == nullptr) {
        throw std::invalid_argument(
            "VulkanBuffer upload data is null"
        );
    }

    if (size > size_) {
        throw std::invalid_argument(
            "VulkanBuffer upload size exceeds buffer size"
        );
    }

    void* mapped = nullptr;

    VkResult result =
        vkMapMemory(
            context_->device(),
            memory_,
            0,
            size,
            0,
            &mapped
        );

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to map Vulkan buffer memory"
        );
    }

    std::memcpy(
        mapped,
        data,
        size
    );

    vkUnmapMemory(
        context_->device(),
        memory_
    );
}

void VulkanBuffer::download(
    void* data,
    std::size_t size
) const {
    if (!valid()) {
        throw std::runtime_error(
            "Cannot download from invalid Vulkan buffer"
        );
    }

    if (data == nullptr) {
        throw std::invalid_argument(
            "VulkanBuffer download data is null"
        );
    }

    if (size > size_) {
        throw std::invalid_argument(
            "VulkanBuffer download size exceeds buffer size"
        );
    }

    void* mapped = nullptr;

    VkResult result =
        vkMapMemory(
            context_->device(),
            memory_,
            0,
            size,
            0,
            &mapped
        );

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to map Vulkan buffer memory"
        );
    }

    std::memcpy(
        data,
        mapped,
        size
    );

    vkUnmapMemory(
        context_->device(),
        memory_
    );
}

void VulkanBuffer::upload_f32(
    const std::vector<float>& data
) {
    if (data.empty()) {
        throw std::invalid_argument(
            "Cannot upload empty float vector"
        );
    }

    upload(
        data.data(),
        data.size() * sizeof(float)
    );
}

std::vector<float> VulkanBuffer::download_f32(
    std::size_t count
) const {
    if (count == 0) {
        throw std::invalid_argument(
            "Cannot download zero floats"
        );
    }

    const std::size_t bytes =
        count * sizeof(float);

    if (bytes > size_) {
        throw std::invalid_argument(
            "Requested float count exceeds buffer size"
        );
    }

    std::vector<float> result(count);

    download(
        result.data(),
        bytes
    );

    return result;
}

uint32_t VulkanBuffer::find_memory_type(
    uint32_t type_filter,
    VkMemoryPropertyFlags properties
) const {
    VkPhysicalDeviceMemoryProperties memory_properties{};

    vkGetPhysicalDeviceMemoryProperties(
        context_->physical_device(),
        &memory_properties
    );

    for (
        uint32_t i = 0;
        i < memory_properties.memoryTypeCount;
        ++i
    ) {
        const bool type_supported =
            (type_filter & (1u << i)) != 0;

        const bool properties_supported =
            (
                memory_properties
                    .memoryTypes[i]
                    .propertyFlags
                & properties
            ) == properties;

        if (
            type_supported &&
            properties_supported
        ) {
            return i;
        }
    }

    throw std::runtime_error(
        "Failed to find suitable Vulkan memory type"
    );
}

void VulkanBuffer::release() {
    if (context_ == nullptr) {
        return;
    }

    VkDevice device =
        context_->device();

    if (device == VK_NULL_HANDLE) {
        return;
    }

    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(
            device,
            buffer_,
            nullptr
        );

        buffer_ = VK_NULL_HANDLE;
    }

    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(
            device,
            memory_,
            nullptr
        );

        memory_ = VK_NULL_HANDLE;
    }

    size_ = 0;
}

} // namespace minitorch
