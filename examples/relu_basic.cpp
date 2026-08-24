#include "minitorch/nn/relu.h"

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

static void expect_close(
    float actual,
    float expected,
    const std::string& message
) {
    if (std::fabs(actual - expected) > 1e-5f) {
        throw std::runtime_error(
            message +
            " expected=" +
            std::to_string(expected) +
            " actual=" +
            std::to_string(actual)
        );
    }
}

static void test_relu_structure() {
    ReLU relu;

    expect(
        relu.name() == "ReLU",
        "ReLU name mismatch"
    );

    expect(
        relu.named_parameters().empty(),
        "ReLU should not have parameters"
    );

    std::cout << "ReLU structure: PASS\n";
}

static void test_relu_forward() {
    ReLU relu;

    Tensor input(
        {
            -2.0f,
            -1.0f,
             0.0f,
             1.0f,
             2.0f
        },
        {
            1, 5
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output =
        relu.forward(input);

    expect(
        output.shape().size() == 2,
        "ReLU output must remain 2D"
    );

    expect(
        output.shape()[0] == 1 &&
        output.shape()[1] == 5,
        "ReLU output shape mismatch"
    );

    expect_close(
        output.item(0),
        0.0f,
        "ReLU[-2]"
    );

    expect_close(
        output.item(1),
        0.0f,
        "ReLU[-1]"
    );

    expect_close(
        output.item(2),
        0.0f,
        "ReLU[0]"
    );

    expect_close(
        output.item(3),
        1.0f,
        "ReLU[1]"
    );

    expect_close(
        output.item(4),
        2.0f,
        "ReLU[2]"
    );

    std::cout << "ReLU forward: PASS\n";
}

static void test_relu_backward() {
    ReLU relu;

    Tensor input(
        {
            -2.0f,
            -1.0f,
             0.0f,
             1.0f,
             2.0f
        },
        {
            1, 5
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output =
        relu.forward(input);

    Tensor loss =
        output.sum();

    loss.backward();

    expect(
        input.has_grad(),
        "ReLU input gradient missing"
    );

    Tensor gradient =
        input.grad();

    expect(
        gradient.size() == 5,
        "ReLU gradient size mismatch"
    );

    expect_close(
        gradient.item(0),
        0.0f,
        "gradient[-2]"
    );

    expect_close(
        gradient.item(1),
        0.0f,
        "gradient[-1]"
    );

    expect_close(
        gradient.item(2),
        0.0f,
        "gradient[0]"
    );

    expect_close(
        gradient.item(3),
        1.0f,
        "gradient[1]"
    );

    expect_close(
        gradient.item(4),
        1.0f,
        "gradient[2]"
    );

    std::cout << "ReLU backward: PASS\n";
}

static void test_relu_preserves_shape() {
    ReLU relu;

    Tensor input(
        {
            -1.0f, 2.0f, -3.0f,
             4.0f, -5.0f, 6.0f
        },
        {
            2, 3
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output =
        relu.forward(input);

    expect(
        output.shape() == input.shape(),
        "ReLU must preserve input shape"
    );

    expect_close(
        output.item(0),
        0.0f,
        "shape test value 0"
    );

    expect_close(
        output.item(1),
        2.0f,
        "shape test value 1"
    );

    expect_close(
        output.item(2),
        0.0f,
        "shape test value 2"
    );

    expect_close(
        output.item(3),
        4.0f,
        "shape test value 3"
    );

    expect_close(
        output.item(4),
        0.0f,
        "shape test value 4"
    );

    expect_close(
        output.item(5),
        6.0f,
        "shape test value 5"
    );

    std::cout << "ReLU shape preservation: PASS\n";
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.3.3 ReLU Test\n"
            << "====================================\n\n";

        test_relu_structure();
        test_relu_forward();
        test_relu_backward();
        test_relu_preserves_shape();

        std::cout
            << "\n====================================\n"
            << " ALL V0.3.3 TESTS PASSED\n"
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
