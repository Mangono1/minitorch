#include "minitorch/vulkan/compute.h"

#include <iostream>
#include <stdexcept>

using namespace minitorch;

int main() {

    try {

        std::cout
            << "====================================\n"
            << " MiniTorch V0.5.2 Vulkan Compute Test\n"
            << "====================================\n\n";

        VulkanContext context;

        if (!context.available()) {
            throw std::runtime_error(
                "Vulkan context unavailable"
            );
        }

        std::cout
            << "Vulkan context: PASS\n";

        VulkanCompute compute(context);

        if (!compute.valid()) {
            throw std::runtime_error(
                "Vulkan compute infrastructure invalid"
            );
        }

        std::cout
            << "Compute infrastructure: PASS\n";

        if (
            compute.command_pool()
            == VK_NULL_HANDLE
        ) {
            throw std::runtime_error(
                "Command pool is null"
            );
        }

        std::cout
            << "Command pool: PASS\n";

        if (
            compute.command_buffer()
            == VK_NULL_HANDLE
        ) {
            throw std::runtime_error(
                "Command buffer is null"
            );
        }

        std::cout
            << "Command buffer: PASS\n";

        if (
            compute.fence()
            == VK_NULL_HANDLE
        ) {
            throw std::runtime_error(
                "Fence is null"
            );
        }

        std::cout
            << "Fence: PASS\n";

        compute.begin();

        std::cout
            << "Command recording: PASS\n";

        compute.end();

        std::cout
            << "Command finalize: PASS\n";

        compute.submit_and_wait();

        std::cout
            << "Queue submit: PASS\n";

        std::cout
            << "Synchronization: PASS\n";

        VulkanCompute moved(
            std::move(compute)
        );

        if (!moved.valid()) {
            throw std::runtime_error(
                "Moved VulkanCompute is invalid"
            );
        }

        std::cout
            << "Move semantics: PASS\n";

        std::cout
            << "\n====================================\n"
            << " ALL V0.5.2 TESTS PASSED\n"
            << "====================================\n";

        return 0;
    }
    catch (const std::exception& error) {

        std::cerr
            << "\nTEST FAILED:\n"
            << error.what()
            << "\n";

        return 1;
    }
}
