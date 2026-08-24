#include "minitorch/optim/sgd.h"

#include <stdexcept>

namespace minitorch {

SGD::SGD(
    const std::vector<Parameter*>& parameters,
    float learning_rate
)
    : parameters_(parameters),
      learning_rate_(learning_rate) {

    if (learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "SGD learning rate must be greater than zero"
        );
    }

    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            throw std::invalid_argument(
                "SGD received a null parameter"
            );
        }
    }
}

void SGD::step() {
    for (Parameter* parameter : parameters_) {
        Tensor& tensor =
            parameter->tensor();

        if (!tensor.requires_grad()) {
            continue;
        }

        if (!tensor.has_grad()) {
            continue;
        }

        Tensor gradient =
            tensor.grad();

        if (gradient.size() != tensor.size()) {
            throw std::runtime_error(
                "SGD gradient size mismatch"
            );
        }

        for (
            std::size_t i = 0;
            i < tensor.size();
            ++i
        ) {
            const float value =
                tensor.item(i);

            const float grad =
                gradient.item(i);

            tensor.set_item(
                i,
                value -
                    learning_rate_ * grad
            );
        }
    }
}

void SGD::zero_grad() {
    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            continue;
        }

        parameter->zero_grad();
    }
}

float SGD::learning_rate() const {
    return learning_rate_;
}

} // namespace minitorch
