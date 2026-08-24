#include "minitorch/nn/mse_loss.h"

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

static void test_structure() {
    MSELoss loss;

    expect(
        loss.name() == "MSELoss",
        "MSELoss name mismatch"
    );

    expect(
        loss.named_parameters().empty(),
        "MSELoss should not have parameters"
    );

    std::cout << "MSELoss structure: PASS\n";
}

static void test_forward() {
    MSELoss loss;

    Tensor prediction(
        {
            2.0f,
            4.0f,
            6.0f,
            8.0f
        },
        {
            1, 4
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor target(
        {
            1.0f,
            2.0f,
            4.0f,
            8.0f
        },
        {
            1, 4
        },
        Device(DeviceType::CPU),
        false
    );

    Tensor output =
        loss.forward(
            prediction,
            target
        );

    expect(
        output.size() == 1,
        "MSELoss output must be scalar"
    );

    /*
        Differences:
        [1, 2, 2, 0]

        Squares:
        [1, 4, 4, 0]

        Mean:
        9 / 4 = 2.25
    */

    expect_close(
        output.item(0),
        2.25f,
        "MSE forward"
    );

    std::cout << "MSELoss forward: PASS\n";
}

static void test_backward() {
    MSELoss loss;

    Tensor prediction(
        {
            2.0f,
            4.0f,
            6.0f,
            8.0f
        },
        {
            1, 4
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor target(
        {
            1.0f,
            2.0f,
            4.0f,
            8.0f
        },
        {
            1, 4
        },
        Device(DeviceType::CPU),
        false
    );

    Tensor output =
        loss.forward(
            prediction,
            target
        );

    output.backward();

    expect(
        prediction.has_grad(),
        "prediction gradient missing"
    );

    Tensor gradient =
        prediction.grad();

    /*
        dMSE/dprediction =
            2 * (prediction - target) / N

        N = 4

        [2*(1)/4,
         2*(2)/4,
         2*(2)/4,
         2*(0)/4]

        = [0.5, 1.0, 1.0, 0.0]
    */

    expect_close(
        gradient.item(0),
        0.5f,
        "MSE gradient[0]"
    );

    expect_close(
        gradient.item(1),
        1.0f,
        "MSE gradient[1]"
    );

    expect_close(
        gradient.item(2),
        1.0f,
        "MSE gradient[2]"
    );

    expect_close(
        gradient.item(3),
        0.0f,
        "MSE gradient[3]"
    );

    std::cout << "MSELoss backward: PASS\n";
}

static void test_shape_validation() {
    MSELoss loss;

    Tensor prediction(
        {
            1.0f,
            2.0f
        },
        {
            1, 2
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor target(
        {
            1.0f,
            2.0f,
            3.0f
        },
        {
            1, 3
        },
        Device(DeviceType::CPU),
        false
    );

    bool threw = false;

    try {
        loss.forward(
            prediction,
            target
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    expect(
        threw,
        "MSELoss should reject mismatched shapes"
    );

    std::cout << "MSELoss validation: PASS\n";
}

static void test_scalar_multiply_gradient() {
    Tensor input(
        {
            2.0f,
            4.0f,
            6.0f
        },
        {
            1, 3
        },
        Device(DeviceType::CPU),
        true
    );

    Tensor output =
        input.multiply_scalar(3.0f);

    Tensor loss =
        output.sum();

    loss.backward();

    expect(
        input.has_grad(),
        "multiply_scalar gradient missing"
    );

    Tensor gradient =
        input.grad();

    expect_close(
        gradient.item(0),
        3.0f,
        "scalar multiply gradient[0]"
    );

    expect_close(
        gradient.item(1),
        3.0f,
        "scalar multiply gradient[1]"
    );

    expect_close(
        gradient.item(2),
        3.0f,
        "scalar multiply gradient[2]"
    );

    std::cout
        << "multiply_scalar backward: PASS\n";
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.4.0 MSE Loss Test\n"
            << "====================================\n\n";

        test_structure();
        test_forward();
        test_backward();
        test_shape_validation();
        test_scalar_multiply_gradient();

        std::cout
            << "\n====================================\n"
            << " ALL V0.4.0 TESTS PASSED\n"
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
