#include "minitorch/nn/linear.h"
#include "minitorch/nn/relu.h"
#include "minitorch/nn/sequential.h"
#include "minitorch/nn/mse_loss.h"
#include "minitorch/optim/sgd.h"

#include <cmath>
#include <iostream>
#include <memory>
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
    float tolerance = 1e-3f
) {
    return std::fabs(a - b) <= tolerance;
}

}

int main() {

    try {

        std::cout
            << "====================================\n"
            << " MiniTorch V0.4.4 MLP Training Test\n"
            << "====================================\n\n";

        /*
         * Model:
         *
         * Linear(1 -> 8)
         * ReLU
         * Linear(8 -> 1)
         */
        auto layer1 =
            std::make_shared<Linear>(1, 8);

        auto activation =
            std::make_shared<ReLU>();

        auto layer2 =
            std::make_shared<Linear>(8, 1);

        Sequential model;

        model.append(layer1);
        model.append(activation);
        model.append(layer2);

        check(
            model.size() == 3,
            "MLP should contain three modules"
        );

        std::cout
            << "MLP structure: PASS\n";

        /*
         * Verify all trainable parameters
         * are visible through Sequential.
         */
        const auto parameters =
            model.parameters();

        check(
            parameters.size() == 4,
            "MLP should contain four parameters"
        );

        std::cout
            << "MLP parameters: PASS\n";

        /*
         * Loss and optimizer.
         */
        MSELoss loss;

        SGD optimizer(
            parameters,
            0.01f
        );

        check(
            close_enough(
                optimizer.learning_rate(),
                0.01f
            ),
            "Unexpected learning rate"
        );

        std::cout
            << "MLP optimizer: PASS\n";

        /*
         * Training dataset:
         *
         * y = 2x
         */
        const std::vector<float> inputs = {
            -2.0f,
            -1.0f,
             0.0f,
             1.0f,
             2.0f
        };

        const std::vector<float> targets = {
            -4.0f,
            -2.0f,
             0.0f,
             2.0f,
             4.0f
        };

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        /*
         * Training.
         */
        for (int epoch = 1; epoch <= 500; ++epoch) {

            float epoch_loss = 0.0f;

            for (
                std::size_t sample = 0;
                sample < inputs.size();
                ++sample
            ) {

                optimizer.zero_grad();

                Tensor input(
                    {inputs[sample]},
                    {1, 1}
                );

                Tensor target(
                    {targets[sample]},
                    {1, 1}
                );

                Tensor prediction =
                    model.forward(input);

                Tensor sample_loss =
                    loss.forward(
                        prediction,
                        target
                    );

                if (epoch == 1) {
                    initial_loss +=
                        sample_loss.item(0);
                }

                epoch_loss +=
                    sample_loss.item(0);

                sample_loss.backward();

                optimizer.step();
            }

            epoch_loss /=
                static_cast<float>(
                    inputs.size()
                );

            if (epoch == 1) {
                initial_loss /=
                    static_cast<float>(
                        inputs.size()
                    );
            }

            if (epoch == 500) {
                final_loss = epoch_loss;
            }

            if (
                epoch == 1 ||
                epoch == 10 ||
                epoch == 100 ||
                epoch == 500
            ) {
                std::cout
                    << "Epoch "
                    << epoch
                    << " | loss = "
                    << epoch_loss
                    << "\n";
            }
        }

        check(
            final_loss < initial_loss,
            "MLP training did not reduce loss"
        );

        std::cout
            << "MLP training: PASS\n";

        /*
         * Final prediction check.
         *
         * Test x = 3.
         *
         * Expected y = 6.
         */
        optimizer.zero_grad();

        Tensor test_input(
            {3.0f},
            {1, 1}
        );

        Tensor test_target(
            {6.0f},
            {1, 1}
        );

        Tensor final_prediction =
            model.forward(test_input);

        Tensor final_test_loss =
            loss.forward(
                final_prediction,
                test_target
            );

        const float prediction =
            final_prediction.item(0);

        std::cout
            << "Final prediction = "
            << prediction
            << "\n";

        std::cout
            << "Final test loss = "
            << final_test_loss.item(0)
            << "\n";

        /*
         * The exact convergence point depends
         * on initialization and optimizer behavior.
         *
         * Require the learned model to be reasonably
         * close to y = 2x.
         */
        check(
            std::fabs(prediction - 6.0f) < 1.0f,
            "MLP failed to learn y = 2x"
        );

        std::cout
            << "Learning target: PASS\n";

        /*
         * Verify gradients can be reset after
         * the final backward pass.
         */
        final_test_loss.backward();

        bool has_gradient = false;

        for (Parameter* parameter : parameters) {
            if (
                parameter != nullptr &&
                parameter->tensor().has_grad()
            ) {
                has_gradient = true;
                break;
            }
        }

        check(
            has_gradient,
            "MLP parameters should receive gradients"
        );

        optimizer.zero_grad();

        for (Parameter* parameter : parameters) {

            if (parameter == nullptr) {
                continue;
            }

            check(
                !parameter->tensor().has_grad(),
                "MLP gradients should be cleared"
            );
        }

        std::cout
            << "MLP gradient reset: PASS\n";

        std::cout
            << "\n====================================\n"
            << " ALL V0.4.4 TESTS PASSED\n"
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
