#pragma once

#include "minitorch/nn/parameter.h"

#include <cstddef>
#include <vector>

namespace minitorch {

class SGD {
public:
    SGD(
        const std::vector<Parameter*>& parameters,
        float learning_rate
    );

    void step();

    void zero_grad();

    float learning_rate() const;

private:
    std::vector<Parameter*> parameters_;
    float learning_rate_;
};

} // namespace minitorch
