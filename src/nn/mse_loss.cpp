#include "minitorch/nn/mse_loss.h"

#include <stdexcept>

namespace minitorch {

MSELoss::MSELoss()
    : Module() {
}

std::string MSELoss::name() const {
    return "MSELoss";
}

Tensor MSELoss::forward(
    const Tensor& prediction,
    const Tensor& target
) const {

    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument(
            "MSELoss requires prediction and target "
            "to have identical shapes"
        );
    }

    if (prediction.size() == 0) {
        throw std::invalid_argument(
            "MSELoss cannot process empty tensors"
        );
    }

    Tensor difference =
        prediction.subtract(target);

    Tensor squared =
        difference.multiply(difference);

    Tensor total =
        squared.sum();

    const float scale =
        1.0f /
        static_cast<float>(prediction.size());

    return total.multiply_scalar(scale);
}

} // namespace minitorch
