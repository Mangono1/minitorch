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
            << " MiniTorch V0.6 Vulkan MUL SCALAR Test\n"
            << "====================================\n\n";

        minitorch::VulkanContext context;

        check(context.available(), "Vulkan context unavailable");

        std::cout
            << "Vulkan context: PASS\n"
            << "Device: "
            << context.device_name()
            << "\n";

        minitorch::VulkanComputePipeline pipeline(
            context,
            "shaders/vector_mul_scalar.comp.spv"
        );

        check(
            pipeline.valid(),
            "Vulkan scalar pipeline invalid"
        );

        std::cout
            << "MUL SCALAR pipeline: PASS\n";

        const std::vector<float> input_data{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        };

        const float scalar = 3.0f;

        const std::vector<float> expected{
            3.0f, 6.0f, 9.0f, 12.0f,
            15.0f, 18.0f, 21.0f, 24.0f
        };

        minitorch::VulkanBuffer input(
            context,
            input_data.size() * sizeof(float)
        );

        minitorch::VulkanBuffer output(
            context,
            input_data.size() * sizeof(float)
        );

        input.upload_f32(input_data);

        std::cout
            << "Input upload: PASS\n";

        minitorch::VulkanCompute compute(context);

        pipeline.dispatch_scalar(
            compute,
            input,
            output,
            input_data.size(),
            scalar
        );

        std::cout
            << "GPU MUL SCALAR dispatch: PASS\n";

        const std::vector<float> result =
            output.download_f32(input_data.size());

        std::cout
            << "Output download: PASS\n";

        for (std::size_t i = 0; i < result.size(); ++i) {
            check(
                close_enough(result[i], expected[i]),
                "MUL SCALAR result mismatch"
            );
        }

        std::cout
            << "MUL SCALAR result: PASS\n\n"
            << "A * "
            << scalar
            << ":\n";

        for (std::size_t i = 0; i < result.size(); ++i) {
            std::cout
                << "  "
                << input_data[i]
                << " * "
                << scalar
                << " = "
                << result[i]
                << "\n";
        }

        std::cout
            << "\n====================================\n"
            << " VULKAN MUL SCALAR TEST PASSED\n"
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
