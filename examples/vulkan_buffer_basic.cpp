#include "minitorch/vulkan/buffer.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace minitorch;

namespace {

void check(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close_enough(
    float a,
    float b,
    float epsilon = 1e-6f
) {
    return std::fabs(a - b) <= epsilon;
}

} // namespace

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.6.0 Vulkan Buffer Test\n"
            << "====================================\n\n";

        VulkanContext context;

        check(
            context.available(),
            "Vulkan context is not available"
        );

        std::cout
            << "Vulkan context: PASS\n";

        check(
            context.device() != VK_NULL_HANDLE,
            "Vulkan logical device is invalid"
        );

        check(
            context.physical_device()
                != VK_NULL_HANDLE,
            "Vulkan physical device is invalid"
        );

        std::cout
            << "Vulkan device: PASS\n";

        const std::vector<float> input = {
            1.0f,
            2.0f,
            3.0f,
            4.0f
        };

        VulkanBuffer buffer(
            context,
            input.size() * sizeof(float)
        );

        check(
            buffer.valid(),
            "Vulkan buffer should be valid"
        );

        check(
            buffer.size()
                == input.size() * sizeof(float),
            "Vulkan buffer size mismatch"
        );

        check(
            buffer.handle() != VK_NULL_HANDLE,
            "Vulkan buffer handle is invalid"
        );

        std::cout
            << "Vulkan buffer creation: PASS\n";

        buffer.upload_f32(input);

        std::cout
            << "Upload CPU -> Vulkan: PASS\n";

        std::vector<float> output =
            buffer.download_f32(input.size());

        check(
            output.size() == input.size(),
            "Downloaded vector size mismatch"
        );

        for (
            std::size_t i = 0;
            i < input.size();
            ++i
        ) {
            check(
                close_enough(
                    input[i],
                    output[i]
                ),
                "Downloaded value mismatch"
            );
        }

        std::cout
            << "Download Vulkan -> CPU: PASS\n";

        std::vector<float> second = {
            10.0f,
            20.0f,
            30.0f,
            40.0f
        };

        buffer.upload_f32(second);

        std::vector<float> second_output =
            buffer.download_f32(second.size());

        for (
            std::size_t i = 0;
            i < second.size();
            ++i
        ) {
            check(
                close_enough(
                    second[i],
                    second_output[i]
                ),
                "Repeated upload/download mismatch"
            );
        }

        std::cout
            << "Repeated upload/download: PASS\n";

        bool oversized_upload_failed = false;

        try {
            std::vector<float> too_large = {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f
            };

            buffer.upload_f32(too_large);
        }
        catch (const std::exception&) {
            oversized_upload_failed = true;
        }

        check(
            oversized_upload_failed,
            "Oversized upload should fail"
        );

        std::cout
            << "Buffer size validation: PASS\n";

        VulkanBuffer moved(
            std::move(buffer)
        );

        check(
            moved.valid(),
            "Moved Vulkan buffer should remain valid"
        );

        check(
            !buffer.valid(),
            "Moved-from Vulkan buffer should be invalid"
        );

        std::vector<float> moved_output =
            moved.download_f32(input.size());

        for (
            std::size_t i = 0;
            i < second.size();
            ++i
        ) {
            check(
                close_enough(
                    second[i],
                    moved_output[i]
                ),
                "Moved buffer data mismatch"
            );
        }

        std::cout
            << "Move semantics: PASS\n";

        std::cout
            << "\n====================================\n"
            << " ALL V0.6.0 TESTS PASSED\n"
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
