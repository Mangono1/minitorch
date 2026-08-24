#pragma once

#include "minitorch/tensor.h"

#include <string>

namespace minitorch {

class Parameter {
public:
    Parameter();

    Parameter(
        const Tensor& tensor,
        const std::string& name = ""
    );

    Tensor& tensor();
    const Tensor& tensor() const;

    const std::string& name() const;
    void set_name(const std::string& name);

    bool requires_grad() const;

    void zero_grad();

private:
    Tensor tensor_;
    std::string name_;
};

} // namespace minitorch
