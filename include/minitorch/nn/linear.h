#pragma once

#include "minitorch/nn/module.h"

#include <cstddef>
#include <string>

namespace minitorch {

class Linear : public Module {
public:
    Linear(
        std::size_t in_features,
        std::size_t out_features,
        bool use_bias = true
    );

    std::string name() const override;

    Tensor forward(
        const Tensor& input
    ) const override;

    std::size_t in_features() const;
    std::size_t out_features() const;

    bool has_bias() const;

private:
    std::size_t in_features_;
    std::size_t out_features_;
    bool use_bias_;
};

} // namespace minitorch
