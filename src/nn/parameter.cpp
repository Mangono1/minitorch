#include "minitorch/nn/parameter.h"

namespace minitorch {

Parameter::Parameter()
    : tensor_(),
      name_() {}

Parameter::Parameter(
    const Tensor& tensor,
    const std::string& name
)
    : tensor_(tensor),
      name_(name) {}

Tensor& Parameter::tensor() {
    return tensor_;
}

const Tensor& Parameter::tensor() const {
    return tensor_;
}

const std::string& Parameter::name() const {
    return name_;
}

void Parameter::set_name(
    const std::string& name
) {
    name_ = name;
}

bool Parameter::requires_grad() const {
    return tensor_.requires_grad();
}

void Parameter::zero_grad() {
    tensor_.zero_grad();
}

} // namespace minitorch
