#include "minitorch/nn/sequential.h"

#include <stdexcept>

namespace minitorch {

Sequential::Sequential()
    : Module(),
      sequence_() {
}

Sequential::Sequential(
    const std::vector<std::shared_ptr<Module>>& modules
)
    : Module(),
      sequence_() {

    for (const auto& module : modules) {
        append(module);
    }
}

std::string Sequential::name() const {
    return "Sequential";
}

void Sequential::append(
    const std::shared_ptr<Module>& module
) {
    if (!module) {
        throw std::invalid_argument(
            "Sequential cannot append a null module"
        );
    }

    const std::string module_name =
        std::to_string(sequence_.size());

    sequence_.push_back(module);

    register_module(
        module_name,
        module
    );
}

Tensor Sequential::forward(
    const Tensor& input
) const {

    Tensor output = input;

    for (const auto& module : sequence_) {
        if (!module) {
            throw std::runtime_error(
                "Sequential contains a null module"
            );
        }

        output =
            module->forward(output);
    }

    return output;
}

std::size_t Sequential::size() const {
    return sequence_.size();
}

Module* Sequential::at(
    std::size_t index
) {
    if (index >= sequence_.size()) {
        throw std::out_of_range(
            "Sequential module index out of range"
        );
    }

    return sequence_[index].get();
}

const Module* Sequential::at(
    std::size_t index
) const {
    if (index >= sequence_.size()) {
        throw std::out_of_range(
            "Sequential module index out of range"
        );
    }

    return sequence_[index].get();
}

} // namespace minitorch
