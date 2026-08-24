#include "minitorch/nn/relu.h"

#include <stdexcept>

namespace minitorch {

ReLU::ReLU()
    : Module() {
}

std::string ReLU::name() const {
    return "ReLU";
}

Tensor ReLU::forward(
    const Tensor& input
) const {

    if (input.size() == 0) {
        throw std::invalid_argument(
            "ReLU cannot process an empty tensor"
        );
    }

    return input.relu();
}

} // namespace minitorch
