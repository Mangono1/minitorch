#include "minitorch/optim/sgd.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

static void test_constructor() {
    Parameter parameter(
        Tensor(
            {
                1.0f,
                2.0f
            },
            {
                1, 2
            },
            Device(DeviceType::CPU),
            true
        )
    );

    std::vector<Parameter*> parameters = {
        &parameter
    };

    SGD optimizer(
        parameters,
        0.1f
    );

    expect_close(
        optimizer.learning_rate(),
        0.1f,
        "SGD learning rate"
    );

    std::cout
        << "SGD constructor: PASS\n";
}

static void test_step() {
    Parameter parameter(
        Tensor(
            {
                1.0f,
                2.0f,
                3.0f
            },
            {
                1, 3
            },
            Device(DeviceType::CPU),
            true
        )
    );

    std::vector<Parameter*> parameters = {
        &parameter
    };

    SGD optimizer(
        parameters,
        0.1f
    );

    Tensor& tensor =
        parameter.tensor();

    Tensor output =
        tensor.sum();

    output.backward();

    expect(
        tensor.has_grad(),
        "Parameter gradient missing"
    );

    Tensor gradient =
        tensor.grad();

    expect_close(
        gradient.item(0),
        1.0f,
        "gradient[0]"
    );

    expect_close(
        gradient.item(1),
        1.0f,
        "gradient[1]"
    );

    expect_close(
        gradient.item(2),
        1.0f,
        "gradient[2]"
    );

    optimizer.step();

    expect_close(
        tensor.item(0),
        0.9f,
        "updated parameter[0]"
    );

    expect_close(
        tensor.item(1),
        1.9f,
        "updated parameter[1]"
    );

    expect_close(
        tensor.item(2),
        2.9f,
        "updated parameter[2]"
    );

    std::cout
        << "SGD step: PASS\n";
}

static void test_zero_grad() {
    Parameter parameter(
        Tensor(
            {
                5.0f,
                6.0f
            },
            {
                1, 2
            },
            Device(DeviceType::CPU),
            true
        )
    );

    std::vector<Parameter*> parameters = {
        &parameter
    };

    SGD optimizer(
        parameters,
        0.01f
    );

    Tensor output =
        parameter.tensor().sum();

    output.backward();

    expect(
        parameter.tensor().has_grad(),
        "Gradient should exist before zero_grad"
    );

    optimizer.zero_grad();

    expect(
        !parameter.tensor().has_grad(),
        "Gradient should be cleared"
    );

    std::cout
        << "SGD zero_grad: PASS\n";
}

static void test_multiple_parameters() {
    Parameter first(
        Tensor(
            {
                10.0f
            },
            {
                1, 1
            },
            Device(DeviceType::CPU),
            true
        )
    );

    Parameter second(
        Tensor(
            {
                20.0f
            },
            {
                1, 1
            },
            Device(DeviceType::CPU),
            true
        )
    );

    std::vector<Parameter*> parameters = {
        &first,
        &second
    };

    SGD optimizer(
        parameters,
        0.5f
    );

    Tensor loss =
        first.tensor().sum()
        .add(second.tensor().sum());

    loss.backward();

    optimizer.step();

    expect_close(
        first.tensor().item(0),
        9.5f,
        "first parameter update"
    );

    expect_close(
        second.tensor().item(0),
        19.5f,
        "second parameter update"
    );

    std::cout
        << "Multiple parameters: PASS\n";
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.4.1 SGD Test\n"
            << "====================================\n\n";

        test_constructor();
        test_step();
        test_zero_grad();
        test_multiple_parameters();

        std::cout
            << "\n====================================\n"
            << " ALL V0.4.1 TESTS PASSED\n"
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
