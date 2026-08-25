#pragma once

#include "minitorch/nn/module.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace minitorch {

class Sequential : public Module {
public:
    Sequential();

    explicit Sequential(
        const std::vector<std::shared_ptr<Module>>& modules
    );

    std::string name() const override;

    void append(
        const std::shared_ptr<Module>& module
    );

    Tensor forward(
        const Tensor& input
    ) const override;

    std::size_t size() const;

    Module* at(
        std::size_t index
    );

    const Module* at(
        std::size_t index
    ) const;

private:
    std::vector<std::shared_ptr<Module>> sequence_;
};

} // namespace minitorch
