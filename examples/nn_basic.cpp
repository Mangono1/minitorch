#include "minitorch/nn/module.h"

#include <iostream>
#include <memory>
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

class TestBlock : public Module {
public:
    TestBlock()
        : Module() {

        Tensor weight(
            {
                1.0f,
                2.0f,
                3.0f,
                4.0f
            },
            {2, 2},
            Device(DeviceType::CPU),
            true
        );

        Tensor bias(
            {
                0.5f,
                1.0f
            },
            {2},
            Device(DeviceType::CPU),
            true
        );

        register_parameter(
            "weight",
            weight
        );

        register_parameter(
            "bias",
            bias
        );
    }

    std::string name() const override {
        return "TestBlock";
    }
};

class TestModel : public Module {
public:
    TestModel()
        : Module() {

        Tensor embedding(
            {
                1.0f,
                2.0f,
                3.0f
            },
            {3},
            Device(DeviceType::CPU),
            true
        );

        register_parameter(
            "embedding",
            embedding
        );

        register_module(
            "block",
            std::make_shared<TestBlock>()
        );
    }

    std::string name() const override {
        return "TestModel";
    }
};

static void test_parameter() {
    Tensor tensor(
        {
            1.0f,
            2.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Parameter parameter(
        tensor,
        "weight"
    );

    expect(
        parameter.requires_grad(),
        "Parameter must require gradients"
    );

    expect(
        parameter.name() == "weight",
        "Parameter name mismatch"
    );

    expect(
        parameter.tensor().size() == 2,
        "Parameter tensor size mismatch"
    );

    std::cout
        << "Parameter: PASS\n";
}

static void test_module_registration() {
    TestBlock block;

    auto parameters =
        block.parameters();

    expect(
        parameters.size() == 2,
        "TestBlock should contain 2 parameters"
    );

    auto named =
        block.named_parameters();

    expect(
        named.size() == 2,
        "TestBlock should contain 2 named parameters"
    );

    bool found_weight = false;
    bool found_bias = false;

    for (const auto& item : named) {
        if (item.first == "weight") {
            found_weight = true;
        }

        if (item.first == "bias") {
            found_bias = true;
        }
    }

    expect(
        found_weight,
        "weight parameter not found"
    );

    expect(
        found_bias,
        "bias parameter not found"
    );

    std::cout
        << "Module registration: PASS\n";
}

static void test_nested_modules() {
    TestModel model;

    auto named =
        model.named_parameters();

    expect(
        named.size() == 3,
        "TestModel should contain 3 parameters"
    );

    bool found_embedding = false;
    bool found_weight = false;
    bool found_bias = false;

    for (const auto& item : named) {
        std::cout
            << "  parameter: "
            << item.first
            << "\n";

        if (item.first == "embedding") {
            found_embedding = true;
        }

        if (item.first == "block.weight") {
            found_weight = true;
        }

        if (item.first == "block.bias") {
            found_bias = true;
        }
    }

    expect(
        found_embedding,
        "embedding parameter not found"
    );

    expect(
        found_weight,
        "nested block.weight not found"
    );

    expect(
        found_bias,
        "nested block.bias not found"
    );

    expect(
        model.child("block") != nullptr,
        "Nested block was not registered"
    );

    std::cout
        << "Nested modules: PASS\n";
}

static void test_zero_grad() {
    Tensor x(
        {
            2.0f,
            3.0f
        },
        {2},
        Device(DeviceType::CPU),
        true
    );

    Tensor loss =
        x.multiply(x).sum();

    loss.backward();

    TestBlock block;

    auto params =
        block.parameters();

    expect(
        params.size() == 2,
        "Unexpected parameter count"
    );

    /*
        Create gradients manually through
        the parameter tensors to verify
        Module::zero_grad().
    */

    Tensor& weight =
        params[0]->tensor();

    Tensor& bias =
        params[1]->tensor();

    Tensor weight_loss =
        weight.sum();

    Tensor bias_loss =
        bias.sum();

    weight_loss.backward();
    bias_loss.backward();

    expect(
        weight.has_grad(),
        "Weight should have gradient"
    );

    expect(
        bias.has_grad(),
        "Bias should have gradient"
    );

    block.zero_grad();

    Tensor weight_grad =
        weight.grad();

    Tensor bias_grad =
        bias.grad();

    for (std::size_t i = 0;
         i < weight_grad.size();
         ++i) {

        expect(
            weight_grad.item(i) == 0.0f,
            "Weight gradient was not cleared"
        );
    }

    for (std::size_t i = 0;
         i < bias_grad.size();
         ++i) {

        expect(
            bias_grad.item(i) == 0.0f,
            "Bias gradient was not cleared"
        );
    }

    std::cout
        << "Module zero_grad: PASS\n";
}

int main() {
    try {
        std::cout
            << "====================================\n";

        std::cout
            << " MiniTorch V0.3.1 NN Foundation Test\n";

        std::cout
            << "====================================\n\n";

        test_parameter();
        test_module_registration();
        test_nested_modules();
        test_zero_grad();

        std::cout
            << "\n====================================\n";

        std::cout
            << " ALL V0.3.1 TESTS PASSED\n";

        std::cout
            << "====================================\n";

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "\nTEST FAILED:\n";

        std::cerr
            << error.what()
            << "\n";

        return 1;
    }
}
