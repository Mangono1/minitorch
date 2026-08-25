#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace minitorch {

struct VulkanDeviceInfo {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;

    std::string name;
    std::string driver_name;

    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;

    std::uint32_t compute_queue_family = UINT32_MAX;

    bool is_cpu = false;
    bool is_gpu = false;
};

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool available() const;

    const std::vector<VulkanDeviceInfo>& devices() const;

    const VulkanDeviceInfo& selected_device() const;

    VkInstance instance() const;

    VkPhysicalDevice physical_device() const;

    VkDevice device() const;

    VkQueue compute_queue() const;

    std::uint32_t compute_queue_family() const;

    std::string device_name() const;

private:
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;

    std::uint32_t compute_queue_family_;

    std::vector<VulkanDeviceInfo> devices_;

    void create_instance();
    void enumerate_devices();
    void select_device();
    void create_logical_device();

    void destroy();

    static bool find_compute_queue_family(
        VkPhysicalDevice physical_device,
        std::uint32_t& family_index
    );
};

} // namespace minitorch
