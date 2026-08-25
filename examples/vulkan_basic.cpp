#include "minitorch/vulkan/context.h"

#include <iostream>
#include <exception>

int main() {

    std::cout
        << "====================================\n"
        << " MiniTorch V0.5.0 Vulkan Test\n"
        << "====================================\n\n";

    try {

        minitorch::VulkanContext context;

        std::cout
            << "Vulkan instance: PASS\n";

        const auto& devices =
            context.devices();

        std::cout
            << "Vulkan devices: "
            << devices.size()
            << "\n";

        if (devices.empty()) {
            throw std::runtime_error(
                "No Vulkan devices available"
            );
        }

        for (
            std::size_t i = 0;
            i < devices.size();
            ++i
        ) {

            const auto& device =
                devices[i];

            std::cout
                << "  Device "
                << i
                << ": "
                << device.name
                << "\n";

            std::cout
                << "    vendor_id = 0x"
                << std::hex
                << device.vendor_id
                << std::dec
                << "\n";

            std::cout
                << "    compute_queue_family = "
                << device.compute_queue_family
                << "\n";

            std::cout
                << "    type = "
                << (
                    device.is_gpu
                        ? "GPU"
                        : device.is_cpu
                            ? "CPU"
                            : "OTHER"
                )
                << "\n";
        }

        if (!context.available()) {
            throw std::runtime_error(
                "Vulkan context is not available"
            );
        }

        std::cout
            << "Vulkan context: PASS\n";

        std::cout
            << "Selected device: "
            << context.device_name()
            << "\n";

        if (
            context.compute_queue()
            == VK_NULL_HANDLE
        ) {
            throw std::runtime_error(
                "Compute queue is null"
            );
        }

        std::cout
            << "Compute queue: PASS\n";

        std::cout
            << "\n====================================\n"
            << " ALL V0.5.0 TESTS PASSED\n"
            << "====================================\n";

        return 0;

    }
    catch (
        const std::exception& error
    ) {

        std::cerr
            << "\nTEST FAILED:\n"
            << error.what()
            << "\n";

        return 1;
    }
}
