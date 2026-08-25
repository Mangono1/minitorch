#pragma once

#include "minitorch/nn/module.h"

namespace minitorch {

class ReLU : public Module {
public:
    ReLU();

    std::string name() const override;

    Tensor forward(
        const Tensor& input
    ) const override;
};

} // namespace minitorch
