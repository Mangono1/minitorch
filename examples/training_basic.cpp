#include "minitorch/nn/linear.h"
#include "minitorch/nn/mse_loss.h"
#include "minitorch/optim/sgd.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace minitorch;

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        std::cout
            << "====================================\n"
            << " MiniTorch V0.4.2 Training Test\n"
            << "====================================\n\n";

        Linear model(1, 1, true);
        MSELoss loss_function;

        std::vector<Parameter*> parameters =
            model.parameters();

        require(
            parameters.size() == 2,
            "Linear should expose weight and bias"
        );

        SGD optimizer(parameters, 0.1f);

        std::cout << "Model parameters: PASS\n";
        std::cout << "Optimizer: PASS\n";

        Tensor input(
            {1.0f},
            {1, 1}
        );

        Tensor target(
            {2.0f},
            {1, 1}
        );

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        constexpr int epochs = 100;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            optimizer.zero_grad();

            Tensor prediction =
                model.forward(input);

            Tensor loss =
                loss_function.forward(
                    prediction,
                    target
                );

            const float loss_value =
                loss.item(0);

            if (epoch == 0) {
                first_loss = loss_value;
            }

            last_loss = loss_value;

            loss.backward();

            optimizer.step();

            if (
                epoch == 0 ||
                epoch == 9 ||
                epoch == 49 ||
                epoch == 99
            ) {
                std::cout
                    << "Epoch "
                    << std::setw(3)
                    << epoch + 1
                    << " | loss = "
                    << std::fixed
                    << std::setprecision(6)
                    << loss_value
                    << "\n";
            }
        }

        require(
            std::isfinite(first_loss),
            "Initial loss is not finite"
        );

        require(
            std::isfinite(last_loss),
            "Final loss is not finite"
        );

        require(
            last_loss < first_loss,
            "Training loss did not decrease"
        );

        std::cout
            << "Training loop: PASS\n";

        optimizer.zero_grad();

        for (Parameter* parameter : parameters) {
            require(
                parameter != nullptr,
                "Null parameter"
            );

            require(
                !parameter->tensor().has_grad(),
                "Gradient should be cleared"
            );
        }

        std::cout
            << "Gradient reset: PASS\n";

        Tensor final_prediction =
            model.forward(input);

        const float predicted =
            final_prediction.item(0);

        require(
            std::isfinite(predicted),
            "Final prediction is not finite"
        );

        std::cout
            << "Final prediction = "
            << std::fixed
            << std::setprecision(6)
            << predicted
            << "\n";

        require(
            std::fabs(predicted - 2.0f) < 0.01f,
            "Model did not learn y = 2x"
        );

        std::cout
            << "Learning target: PASS\n"
            << "\n====================================\n"
            << " ALL V0.4.2 TESTS PASSED\n"
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
