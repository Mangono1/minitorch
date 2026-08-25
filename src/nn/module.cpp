#include "minitorch/nn/module.h"

#include <stdexcept>

namespace minitorch {

Module::Module()
    : parameters_(),
      modules_() {
}

std::string Module::name() const {
    return "Module";
}

Tensor Module::forward(
    const Tensor&
) const {
    throw std::runtime_error(
        "Module::forward() is not implemented"
    );
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

    parameters_.insert_or_assign(
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

    parameters_.insert_or_assign(
        name,
        Parameter(
            parameter.tensor(),
            name
        )
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
            "Cannot register a null module"
        );
    }

    modules_.insert_or_assign(
        name,
        std::move(module)
    );
}

std::vector<Parameter*> Module::parameters() {
    std::vector<Parameter*> result;

    for (auto& entry : parameters_) {
        result.push_back(&entry.second);
    }

    for (auto& entry : modules_) {
        if (entry.second) {
            auto child_parameters =
                entry.second->parameters();

            result.insert(
                result.end(),
                child_parameters.begin(),
                child_parameters.end()
            );
        }
    }

    return result;
}

std::vector<const Parameter*> Module::parameters() const {
    std::vector<const Parameter*> result;

    for (const auto& entry : parameters_) {
        result.push_back(&entry.second);
    }

    for (const auto& entry : modules_) {
        if (entry.second) {
            auto child_parameters =
                entry.second->parameters();

            result.insert(
                result.end(),
                child_parameters.begin(),
                child_parameters.end()
            );
        }
    }

    return result;
}

std::vector<
    std::pair<std::string, Parameter*>
> Module::named_parameters() {

    std::vector<
        std::pair<std::string, Parameter*>
    > result;

    collect_named_parameters(
        "",
        result
    );

    return result;
}

std::vector<
    std::pair<std::string, const Parameter*>
> Module::named_parameters() const {

    std::vector<
        std::pair<std::string, const Parameter*>
    > result;

    collect_named_parameters(
        "",
        result
    );

    return result;
}

void Module::collect_named_parameters(
    const std::string& prefix,
    std::vector<
        std::pair<std::string, Parameter*>
    >& output
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
        if (!entry.second) {
            continue;
        }

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
    std::vector<
        std::pair<std::string, const Parameter*>
    >& output
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
        if (!entry.second) {
            continue;
        }

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

void Module::zero_grad() {
    for (auto& entry : parameters_) {
        entry.second.zero_grad();
    }

    for (auto& entry : modules_) {
        if (entry.second) {
            entry.second->zero_grad();
        }
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
