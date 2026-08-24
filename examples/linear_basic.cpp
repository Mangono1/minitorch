#include "minitorch/nn/linear.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace minitorch;

static void expect(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_linear_structure() {
    Linear layer(3, 2);

    expect(
        layer.in_features() == 3,
        "in_features mismatch"
    );

    expect(
        layer.out_features() == 2,
        "out_features mismatch"
    );

    expect(
        layer.has_bias(),
        "Linear should have bias"
    );

    auto named = layer.named_parameters();

    expect(
        named.size() == 2,
        "Linear should contain weight and bias"
    );

    bool found_weight = false;
    bool found_bias = false;

    for (const auto& item : named) {
        std::cout << "  parameter: "
                  << item.first
                  << "\n";

        if (item.first == "weight") {
            found_weight = true;

            const auto& shape =
                item.second->tensor().shape();

            expect(
                shape.size() == 2,
                "weight must be 2D"
            );

            expect(
                shape[0] == 3 &&
                shape[1] == 2,
                "weight shape must be [3, 2]"
            );
        }

        if (item.first == "bias") {
            found_bias = true;

            const auto& shape =
                item.second->tensor().shape();

            expect(
                shape.size() == 2,
                "bias must be 2D"
            );

            expect(
                shape[0] == 1 &&
                shape[1] == 2,
                "bias shape must be [1, 2]"
            );
        }
    }

    expect(found_weight, "weight not found");
    expect(found_bias, "bias not found");

    std::cout << "Linear structure: PASS\n";
}

static void test_linear_forward() {
    Linear layer(3, 2);

    Tensor input(
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        },
        {
            2, 3
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output = layer.forward(input);

    const auto& shape = output.shape();

    expect(
        shape.size() == 2,
        "output must be 2D"
    );

    expect(
        shape[0] == 2 &&
        shape[1] == 2,
        "output shape must be [2, 2]"
    );

    expect(
        output.size() == 4,
        "output size must be 4"
    );

    std::cout << "Linear forward: PASS\n";
}

static void test_linear_backward() {
    Linear layer(3, 2);

    Tensor input(
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        },
        {
            2, 3
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output = layer.forward(input);

    Tensor loss = output.sum();

    expect(
        loss.requires_grad(),
        "loss should require gradient"
    );

    loss.backward();

    auto named = layer.named_parameters();

    for (auto& item : named) {
        Parameter* parameter = item.second;

        expect(
            parameter->tensor().has_grad(),
            "missing gradient for parameter: " +
            item.first
        );

        Tensor gradient =
            parameter->tensor().grad();

        expect(
            gradient.size() ==
            parameter->tensor().size(),
            "gradient size mismatch for: " +
            item.first
        );

        std::cout
            << "  gradient: "
            << item.first
            << " size="
            << gradient.size()
            << "\n";
    }

    expect(
        input.has_grad(),
        "input gradient missing"
    );

    expect(
        input.grad().size() == input.size(),
        "input gradient size mismatch"
    );

    std::cout << "Linear backward: PASS\n";
}

static void test_linear_validation() {
    Linear layer(3, 2);

    bool threw = false;

    try {
        Tensor invalid(
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            },
            {
                2, 2
            },
            Device(DeviceType::CPU),
            true
        );

        layer.forward(invalid);

    } catch (const std::invalid_argument&) {
        threw = true;
    }

    expect(
        threw,
        "Linear should reject invalid input feature count"
    );

    std::cout << "Linear validation: PASS\n";
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.3.2 Linear Test\n"
            << "====================================\n\n";

        test_linear_structure();
        test_linear_forward();
        test_linear_backward();
        test_linear_validation();

        std::cout
            << "\n====================================\n"
            << " ALL V0.3.2 TESTS PASSED\n"
            << "====================================\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "\nTEST FAILED:\n"
            << error.what()
            << "\n";

        return 1;
    }
}
