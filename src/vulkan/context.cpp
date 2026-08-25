#include "minitorch/vulkan/context.h"

#include <stdexcept>
#include <sstream>

namespace minitorch {

VulkanContext::VulkanContext()
    : instance_(VK_NULL_HANDLE),
      physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE),
      compute_queue_(VK_NULL_HANDLE),
      compute_queue_family_(UINT32_MAX),
      devices_() {

    create_instance();
    enumerate_devices();
    select_device();
    create_logical_device();
}

VulkanContext::~VulkanContext() {
    destroy();
}

bool VulkanContext::available() const {
    return instance_ != VK_NULL_HANDLE &&
           physical_device_ != VK_NULL_HANDLE &&
           device_ != VK_NULL_HANDLE &&
           compute_queue_ != VK_NULL_HANDLE;
}

const std::vector<VulkanDeviceInfo>&
VulkanContext::devices() const {
    return devices_;
}

const VulkanDeviceInfo&
VulkanContext::selected_device() const {
    if (physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "No Vulkan physical device selected"
        );
    }

    for (const auto& info : devices_) {
        if (info.physical_device == physical_device_) {
            return info;
        }
    }

    throw std::runtime_error(
        "Selected Vulkan device information not found"
    );
}

VkInstance VulkanContext::instance() const {
    return instance_;
}

VkPhysicalDevice VulkanContext::physical_device() const {
    return physical_device_;
}

VkDevice VulkanContext::device() const {
    return device_;
}

VkQueue VulkanContext::compute_queue() const {
    return compute_queue_;
}

std::uint32_t VulkanContext::compute_queue_family() const {
    return compute_queue_family_;
}

std::string VulkanContext::device_name() const {
    if (physical_device_ == VK_NULL_HANDLE) {
        return "none";
    }

    return selected_device().name;
}

void VulkanContext::create_instance() {

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MiniTorch";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 5, 0);
    app_info.pEngineName = "MiniTorch";
    app_info.engineVersion = VK_MAKE_VERSION(0, 5, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType =
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    create_info.pApplicationInfo = &app_info;

    const VkResult result =
        vkCreateInstance(
            &create_info,
            nullptr,
            &instance_
        );

    if (result != VK_SUCCESS) {
        std::ostringstream message;

        message
            << "Failed to create Vulkan instance. VkResult="
            << static_cast<int>(result);

        throw std::runtime_error(message.str());
    }
}

bool VulkanContext::find_compute_queue_family(
    VkPhysicalDevice physical_device,
    std::uint32_t& family_index
) {
    std::uint32_t count = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device,
        &count,
        nullptr
    );

    if (count == 0) {
        return false;
    }

    std::vector<VkQueueFamilyProperties> families(
        count
    );

    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device,
        &count,
        families.data()
    );

    for (std::uint32_t i = 0; i < count; ++i) {

        const VkQueueFlags flags =
            families[i].queueFlags;

        if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
            family_index = i;
            return true;
        }
    }

    return false;
}

void VulkanContext::enumerate_devices() {

    std::uint32_t count = 0;

    VkResult result =
        vkEnumeratePhysicalDevices(
            instance_,
            &count,
            nullptr
        );

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to enumerate Vulkan physical devices"
        );
    }

    if (count == 0) {
        throw std::runtime_error(
            "No Vulkan physical devices found"
        );
    }

    std::vector<VkPhysicalDevice> physical_devices(
        count
    );

    result =
        vkEnumeratePhysicalDevices(
            instance_,
            &count,
            physical_devices.data()
        );

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to retrieve Vulkan physical devices"
        );
    }

    devices_.clear();

    for (VkPhysicalDevice physical_device :
         physical_devices) {

        VkPhysicalDeviceProperties properties{};

        vkGetPhysicalDeviceProperties(
            physical_device,
            &properties
        );

        std::uint32_t compute_family =
            UINT32_MAX;

        const bool has_compute =
            find_compute_queue_family(
                physical_device,
                compute_family
            );

        if (!has_compute) {
            continue;
        }

        VulkanDeviceInfo info;

        info.physical_device =
            physical_device;

        info.name =
            properties.deviceName;

        info.vendor_id =
            properties.vendorID;

        info.device_id =
            properties.deviceID;

        info.compute_queue_family =
            compute_family;

        switch (properties.deviceType) {

            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                info.is_gpu = true;
                break;

            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                info.is_gpu = true;
                break;

            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                info.is_cpu = true;
                break;

            default:
                break;
        }

        devices_.push_back(info);
    }

    if (devices_.empty()) {
        throw std::runtime_error(
            "No Vulkan device with compute queue found"
        );
    }
}

void VulkanContext::select_device() {

    /*
     * Prefer a real GPU over a CPU Vulkan implementation.
     */

    for (const auto& info : devices_) {

        if (info.is_gpu) {
            physical_device_ =
                info.physical_device;

            compute_queue_family_ =
                info.compute_queue_family;

            return;
        }
    }

    /*
     * Fallback to CPU Vulkan implementation,
     * such as llvmpipe.
     */

    for (const auto& info : devices_) {

        if (info.is_cpu) {
            physical_device_ =
                info.physical_device;

            compute_queue_family_ =
                info.compute_queue_family;

            return;
        }
    }

    throw std::runtime_error(
        "Unable to select a Vulkan device"
    );
}

void VulkanContext::create_logical_device() {

    if (physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "Cannot create Vulkan device without physical device"
        );
    }

    const float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info{};

    queue_info.sType =
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;

    queue_info.queueFamilyIndex =
        compute_queue_family_;

    queue_info.queueCount = 1;

    queue_info.pQueuePriorities =
        &priority;

    VkDeviceCreateInfo device_info{};

    device_info.sType =
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    device_info.queueCreateInfoCount = 1;

    device_info.pQueueCreateInfos =
        &queue_info;

    const VkResult result =
        vkCreateDevice(
            physical_device_,
            &device_info,
            nullptr,
            &device_
        );

    if (result != VK_SUCCESS) {

        std::ostringstream message;

        message
            << "Failed to create Vulkan logical device. "
            << "VkResult="
            << static_cast<int>(result);

        throw std::runtime_error(
            message.str()
        );
    }

    vkGetDeviceQueue(
        device_,
        compute_queue_family_,
        0,
        &compute_queue_
    );

    if (compute_queue_ == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "Failed to obtain Vulkan compute queue"
        );
    }
}

void VulkanContext::destroy() {

    if (device_ != VK_NULL_HANDLE) {

        vkDeviceWaitIdle(device_);

        vkDestroyDevice(
            device_,
            nullptr
        );

        device_ =
            VK_NULL_HANDLE;

        compute_queue_ =
            VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {

        vkDestroyInstance(
            instance_,
            nullptr
        );

        instance_ =
            VK_NULL_HANDLE;
    }

    physical_device_ =
        VK_NULL_HANDLE;

    compute_queue_family_ =
        UINT32_MAX;
}

} // namespace minitorch
