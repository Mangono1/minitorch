#include "minitorch/nn/linear.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace minitorch {

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features,
    bool use_bias
)
    : Module(),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {

    if (in_features == 0) {
        throw std::invalid_argument(
            "Linear in_features cannot be zero"
        );
    }

    if (out_features == 0) {
        throw std::invalid_argument(
            "Linear out_features cannot be zero"
        );
    }

    /*
        Xavier-style initialization.

        limit = sqrt(6 / (in + out))
    */

    const float limit =
        std::sqrt(
            6.0f /
            static_cast<float>(
                in_features + out_features
            )
        );

    std::mt19937 generator(
        1234567
    );

    std::uniform_real_distribution<float>
        distribution(
            -limit,
            limit
        );

    std::vector<float> weight_data(
        in_features * out_features
    );

    for (float& value : weight_data) {
        value = distribution(generator);
    }

    Tensor weight(
        weight_data,
        {
            in_features,
            out_features
        },
        Device(DeviceType::CPU),
        true
    );

    register_parameter(
        "weight",
        weight
    );

    if (use_bias_) {

        std::vector<float> bias_data(
            out_features,
            0.0f
        );

        Tensor bias(
            bias_data,
            {
                1,
                out_features
            },
            Device(DeviceType::CPU),
            true
        );

        register_parameter(
            "bias",
            bias
        );
    }
}

std::string Linear::name() const {
    return "Linear";
}

Tensor Linear::forward(
    const Tensor& input
) const {

    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "Linear currently expects a 2D input"
        );
    }

    if (
        input.shape()[1] !=
        in_features_
    ) {
        throw std::invalid_argument(
            "Linear input feature count mismatch"
        );
    }

    const auto weight_it =
        parameters_.find("weight");

    if (weight_it == parameters_.end()) {
        throw std::runtime_error(
            "Linear weight parameter is missing"
        );
    }

    Tensor output =
        input.matmul(
            weight_it->second.tensor()
        );

    if (use_bias_) {

        const auto bias_it =
            parameters_.find("bias");

        if (
            bias_it ==
            parameters_.end()
        ) {
            throw std::runtime_error(
                "Linear bias parameter is missing"
            );
        }

        output =
            output.add_bias_2d(
                bias_it->second.tensor()
            );
    }

    return output;
}

std::size_t Linear::in_features() const {
    return in_features_;
}

std::size_t Linear::out_features() const {
    return out_features_;
}

bool Linear::has_bias() const {
    return use_bias_;
}

} // namespace minitorch
