#pragma once

#include "minitorch/nn/module.h"

namespace minitorch {

class MSELoss : public Module {
public:
    MSELoss();

    std::string name() const override;

    Tensor forward(
        const Tensor& prediction,
        const Tensor& target
    ) const;
};

} // namespace minitorch
