#include "minitorch/device.h"

namespace minitorch {

Device::Device()
    : type_(DeviceType::CPU) {}

Device::Device(DeviceType type)
    : type_(type) {}

DeviceType Device::type() const {
    return type_;
}

std::string Device::name() const {
    switch (type_) {
        case DeviceType::CPU:
            return "cpu";

        case DeviceType::Vulkan:
            return "vulkan";

        default:
            return "unknown";
    }
}

bool Device::is_cpu() const {
    return type_ == DeviceType::CPU;
}

bool Device::is_vulkan() const {
    return type_ == DeviceType::Vulkan;
}

} // namespace minitorch
