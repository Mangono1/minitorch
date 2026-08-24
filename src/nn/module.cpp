#include "minitorch/nn/module.h"

#include <stdexcept>

namespace minitorch {

Module::Module() = default;

std::string Module::name() const {
    return "Module";
}

void Module::register_parameter(
    const std::string& name,
    const Tensor& tensor
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Parameter name cannot be empty"
        );
    }

    if (!tensor.requires_grad()) {
        throw std::invalid_argument(
            "Registered parameter must require gradients"
        );
    }

    if (parameters_.count(name)) {
        throw std::invalid_argument(
            "Parameter already registered: " + name
        );
    }

    if (modules_.count(name)) {
        throw std::invalid_argument(
            "A module with this name is already registered: " + name
        );
    }

    parameters_.emplace(
        name,
        Parameter(tensor, name)
    );
}

void Module::register_parameter(
    const std::string& name,
    const Parameter& parameter
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Parameter name cannot be empty"
        );
    }

    if (!parameter.requires_grad()) {
        throw std::invalid_argument(
            "Registered parameter must require gradients"
        );
    }

    if (parameters_.count(name)) {
        throw std::invalid_argument(
            "Parameter already registered: " + name
        );
    }

    if (modules_.count(name)) {
        throw std::invalid_argument(
            "A module with this name is already registered: " + name
        );
    }

    parameters_.emplace(
        name,
        Parameter(parameter.tensor(), name)
    );
}

void Module::register_module(
    const std::string& name,
    std::shared_ptr<Module> module
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Module name cannot be empty"
        );
    }

    if (!module) {
        throw std::invalid_argument(
            "Cannot register null module"
        );
    }

    if (modules_.count(name)) {
        throw std::invalid_argument(
            "Module already registered: " + name
        );
    }

    if (parameters_.count(name)) {
        throw std::invalid_argument(
            "A parameter with this name is already registered: " + name
        );
    }

    modules_.emplace(
        name,
        std::move(module)
    );
}

void Module::collect_named_parameters(
    const std::string& prefix,
    std::vector<std::pair<std::string, Parameter*>>& output
) {
    for (auto& entry : parameters_) {
        const std::string full_name =
            prefix.empty()
                ? entry.first
                : prefix + "." + entry.first;

        output.emplace_back(
            full_name,
            &entry.second
        );
    }

    for (auto& entry : modules_) {
        const std::string child_prefix =
            prefix.empty()
                ? entry.first
                : prefix + "." + entry.first;

        entry.second->collect_named_parameters(
            child_prefix,
            output
        );
    }
}

void Module::collect_named_parameters(
    const std::string& prefix,
    std::vector<std::pair<std::string, const Parameter*>>& output
) const {
    for (const auto& entry : parameters_) {
        const std::string full_name =
            prefix.empty()
                ? entry.first
                : prefix + "." + entry.first;

        output.emplace_back(
            full_name,
            &entry.second
        );
    }

    for (const auto& entry : modules_) {
        const std::string child_prefix =
            prefix.empty()
                ? entry.first
                : prefix + "." + entry.first;

        entry.second->collect_named_parameters(
            child_prefix,
            output
        );
    }
}

std::vector<Parameter*> Module::parameters() {
    std::vector<Parameter*> result;

    auto named = named_parameters();

    for (auto& item : named) {
        result.push_back(item.second);
    }

    return result;
}

std::vector<const Parameter*> Module::parameters() const {
    std::vector<const Parameter*> result;

    auto named = named_parameters();

    for (const auto& item : named) {
        result.push_back(item.second);
    }

    return result;
}

std::vector<std::pair<std::string, Parameter*>> Module::named_parameters() {
    std::vector<std::pair<std::string, Parameter*>> result;

    collect_named_parameters(
        "",
        result
    );

    return result;
}

std::vector<std::pair<std::string, const Parameter*>> Module::named_parameters() const {
    std::vector<std::pair<std::string, const Parameter*>> result;

    collect_named_parameters(
        "",
        result
    );

    return result;
}

void Module::zero_grad() {
    for (auto& entry : parameters_) {
        entry.second.zero_grad();
    }

    for (auto& entry : modules_) {
        entry.second->zero_grad();
    }
}

Module* Module::child(
    const std::string& name
) {
    auto it = modules_.find(name);

    if (it == modules_.end()) {
        return nullptr;
    }

    return it->second.get();
}

const Module* Module::child(
    const std::string& name
) const {
    auto it = modules_.find(name);

    if (it == modules_.end()) {
        return nullptr;
    }

    return it->second.get();
}

} // namespace minitorch
