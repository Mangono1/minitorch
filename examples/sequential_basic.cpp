#include "minitorch/nn/sequential.h"
#include "minitorch/nn/linear.h"
#include "minitorch/nn/relu.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitorch;

static void check(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static bool close_enough(
    float a,
    float b,
    float tolerance = 1e-5f
) {
    return std::fabs(a - b) <= tolerance;
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.4.3 Sequential Test\n"
            << "====================================\n\n";

        /*
         * Model:
         *
         * Linear(2 -> 3)
         *      |
         *     ReLU
         *      |
         * Linear(3 -> 1)
         */

        auto linear1 =
            std::make_shared<Linear>(2, 3, true);

        auto relu =
            std::make_shared<ReLU>();

        auto linear2 =
            std::make_shared<Linear>(3, 1, true);

        Sequential model;

        model.append(linear1);
        model.append(relu);
        model.append(linear2);

        check(
            model.size() == 3,
            "Sequential size should be 3"
        );

        check(
            model.at(0) != nullptr,
            "Sequential first module missing"
        );

        check(
            model.at(1) != nullptr,
            "Sequential second module missing"
        );

        check(
            model.at(2) != nullptr,
            "Sequential third module missing"
        );

        std::cout
            << "Sequential structure: PASS\n";

        /*
         * Parameter registration.
         *
         * Linear 1:
         *   weight = 6
         *   bias   = 3
         *
         * Linear 2:
         *   weight = 3
         *   bias   = 1
         *
         * Total = 13 parameters
         */

        auto parameters =
            model.parameters();

        check(
            parameters.size() == 4,
            "Sequential should expose four parameters"
        );

        std::cout
            << "Parameter registration: PASS\n";

        for (const auto& item :
             model.named_parameters()) {

            std::cout
                << "  parameter: "
                << item.first
                << "\n";
        }

        /*
         * Input:
         *
         * [1, -2]
         *
         * Shape = [1, 2]
         */

        Tensor input(
            {1.0f, -2.0f},
            {1, 2},
            Device(DeviceType::CPU),
            false
        );

        Tensor output =
            model.forward(input);

        check(
            output.shape()
                == std::vector<std::size_t>{1, 1},
            "Sequential output shape mismatch"
        );

        std::cout
            << "Sequential forward: PASS\n";

        /*
         * Backward test.
         *
         * We only need to prove that gradients propagate
         * through the entire Sequential graph.
         */

        Tensor loss =
            output.sum();

        loss.backward();

        bool found_gradient = false;

        for (Parameter* parameter :
             parameters) {

            if (parameter == nullptr) {
                continue;
            }

            Tensor& tensor =
                parameter->tensor();

            if (tensor.has_grad()) {
                found_gradient = true;

                check(
                    tensor.grad().size()
                        == tensor.size(),
                    "Gradient size mismatch"
                );
            }
        }

        check(
            found_gradient,
            "No parameter received gradient"
        );

        std::cout
            << "Sequential backward: PASS\n";

        /*
         * zero_grad
         */

        model.zero_grad();

        for (Parameter* parameter :
             parameters) {

            check(
                parameter != nullptr,
                "Null parameter detected"
            );

            Tensor& tensor =
                parameter->tensor();

            if (tensor.requires_grad()) {
                Tensor gradient =
                    tensor.grad();

                check(
                    gradient.size() == tensor.size(),
                    "Gradient size mismatch after zero_grad"
                );

                for (
                    std::size_t i = 0;
                    i < gradient.size();
                    ++i
                ) {
                    check(
                        close_enough(
                            gradient.item(i),
                            0.0f
                        ),
                        "Gradient should be zero"
                    );
                }
            }
        }

        std::cout
            << "Sequential zero_grad: PASS\n";

        /*
         * Forward again after zero_grad.
         *
         * This verifies that clearing gradients does not
         * destroy model parameters or the forward path.
         */

        Tensor output2 =
            model.forward(input);

        check(
            output2.shape()
                == std::vector<std::size_t>{1, 1},
            "Second forward shape mismatch"
        );

        std::cout
            << "Sequential repeat forward: PASS\n";

        /*
         * Empty Sequential validation.
         */

        Sequential empty;

        check(
            empty.size() == 0,
            "Empty Sequential should have size zero"
        );

        Tensor empty_output =
            empty.forward(input);

        check(
            empty_output.shape() == input.shape(),
            "Empty Sequential should preserve input shape"
        );

        check(
            empty_output.size() == input.size(),
            "Empty Sequential should preserve input size"
        );

        for (
            std::size_t i = 0;
            i < input.size();
            ++i
        ) {
            check(
                close_enough(
                    empty_output.item(i),
                    input.item(i)
                ),
                "Empty Sequential should behave as identity"
            );
        }

        std::cout
            << "Sequential empty identity: PASS\n";

        std::cout
            << "\n====================================\n"
            << " ALL V0.4.3 TESTS PASSED\n"
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
