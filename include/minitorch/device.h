#pragma once

#include <string>

namespace minitorch {

enum class DeviceType {
    CPU,
    Vulkan
};

class Device {
public:
    Device();
    explicit Device(DeviceType type);

    DeviceType type() const;
    std::string name() const;
    bool is_cpu() const;
    bool is_vulkan() const;

private:
    DeviceType type_;
};

} // namespace minitorch
