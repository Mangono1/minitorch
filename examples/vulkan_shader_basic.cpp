#include "minitorch/vulkan/buffer.h"
#include "minitorch/vulkan/compute.h"
#include "minitorch/vulkan/context.h"
#include "minitorch/vulkan/pipeline.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void check(
    bool condition,
    const char* message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close_enough(
    float a,
    float b,
    float epsilon = 1e-4f
) {
    return std::fabs(a - b) <= epsilon;
}

} // namespace

int main() {

    try {

        std::cout
            << "====================================\n"
            << " MiniTorch V0.5.3 Vulkan Shader Test\n"
            << "====================================\n\n";

        minitorch::VulkanContext context;

        check(
            context.available(),
            "Vulkan context unavailable"
        );

        std::cout
            << "Vulkan context: PASS\n";

        minitorch::VulkanComputePipeline pipeline(
            context,
            "shaders/vector_mul2.comp.spv"
        );

        check(
            pipeline.valid(),
            "Vulkan compute pipeline invalid"
        );

        std::cout
            << "Shader module: PASS\n"
            << "Descriptor layout: PASS\n"
            << "Pipeline layout: PASS\n"
            << "Compute pipeline: PASS\n";

        const std::vector<float> input_data{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            7.0f,
            8.0f
        };

        const std::vector<float> expected{
            2.0f,
            4.0f,
            6.0f,
            8.0f,
            10.0f,
            12.0f,
            14.0f,
            16.0f
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

        minitorch::VulkanCompute compute(
            context
        );

        pipeline.dispatch(
            compute,
            input,
            output,
            input_data.size()
        );

        std::cout
            << "Compute dispatch: PASS\n"
            << "Queue synchronization: PASS\n";

        const std::vector<float> result =
            output.download_f32(
                input_data.size()
            );

        bool correct = true;

        for (
            std::size_t i = 0;
            i < result.size();
            ++i
        ) {
            if (
                !close_enough(
                    result[i],
                    expected[i]
                )
            ) {
                correct = false;
            }
        }

        check(
            correct,
            "Compute result mismatch"
        );

        std::cout
            << "Download output: PASS\n"
            << "Compute result: PASS\n";

        std::cout
            << "\nInput:\n";

        for (float value : input_data) {
            std::cout
                << "  "
                << value
                << "\n";
        }

        std::cout
            << "\nOutput:\n";

        for (float value : result) {
            std::cout
                << "  "
                << value
                << "\n";
        }

        std::cout
            << "\n====================================\n"
            << " ALL V0.5.3 TESTS PASSED\n"
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
