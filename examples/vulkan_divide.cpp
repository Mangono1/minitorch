#include "minitorch/vulkan/buffer.h"
#include "minitorch/vulkan/compute.h"
#include "minitorch/vulkan/context.h"
#include "minitorch/vulkan/pipeline.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close_enough(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) <= epsilon;
}

}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.6 Vulkan DIVIDE Test\n"
            << "====================================\n\n";

        minitorch::VulkanContext context;

        check(context.available(), "Vulkan context unavailable");

        std::cout << "Vulkan context: PASS\n";
        std::cout
            << "Device: "
            << context.device_name()
            << "\n";

        minitorch::VulkanComputePipeline pipeline(
            context,
            "shaders/vector_divide.comp.spv",
            3
        );

        check(
            pipeline.valid(),
            "Vulkan divide pipeline invalid"
        );

        std::cout << "DIVIDE pipeline: PASS\n";

        const std::vector<float> a{
            10.0f, 20.0f, 30.0f, 40.0f,
            50.0f, 60.0f, 70.0f, 80.0f
        };

        const std::vector<float> b{
            2.0f, 4.0f, 5.0f, 8.0f,
            10.0f, 12.0f, 14.0f, 16.0f
        };

        const std::vector<float> expected{
            5.0f, 5.0f, 6.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f
        };

        minitorch::VulkanBuffer input_a(
            context,
            a.size() * sizeof(float)
        );

        minitorch::VulkanBuffer input_b(
            context,
            b.size() * sizeof(float)
        );

        minitorch::VulkanBuffer output(
            context,
            a.size() * sizeof(float)
        );

        input_a.upload_f32(a);
        input_b.upload_f32(b);

        std::cout << "Input upload: PASS\n";

        minitorch::VulkanCompute compute(context);

        std::vector<minitorch::VulkanBuffer*> buffers{
            &input_a,
            &input_b,
            &output
        };

        pipeline.dispatch(
            compute,
            buffers,
            a.size()
        );

        std::cout
            << "GPU DIVIDE dispatch: PASS\n";

        const std::vector<float> result =
            output.download_f32(a.size());

        std::cout
            << "Output download: PASS\n";

        for (std::size_t i = 0; i < result.size(); ++i) {
            check(
                close_enough(result[i], expected[i]),
                "DIVIDE result mismatch"
            );
        }

        std::cout
            << "DIVIDE result: PASS\n\n"
            << "A / B:\n";

        for (std::size_t i = 0; i < result.size(); ++i) {
            std::cout
                << "  "
                << a[i]
                << " / "
                << b[i]
                << " = "
                << result[i]
                << "\n";
        }

        std::cout
            << "\n====================================\n"
            << " VULKAN DIVIDE TEST PASSED\n"
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
