#pragma once

#include "minitorch/nn/parameter.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace minitorch {

class Module {
public:
    Module();
    virtual ~Module() = default;

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    virtual std::string name() const;

    virtual Tensor forward(const Tensor& input) const;

    void register_parameter(
        const std::string& name,
        const Tensor& tensor
    );

    void register_parameter(
        const std::string& name,
        const Parameter& parameter
    );

    void register_module(
        const std::string& name,
        std::shared_ptr<Module> module
    );

    std::vector<Parameter*> parameters();
    std::vector<const Parameter*> parameters() const;

    std::vector<
        std::pair<std::string, Parameter*>
    > named_parameters();

    std::vector<
        std::pair<std::string, const Parameter*>
    > named_parameters() const;

    void zero_grad();

    Module* child(
        const std::string& name
    );

    const Module* child(
        const std::string& name
    ) const;

protected:
    std::map<std::string, Parameter> parameters_;
    std::map<std::string, std::shared_ptr<Module>> modules_;

private:
    void collect_named_parameters(
        const std::string& prefix,
        std::vector<
            std::pair<std::string, Parameter*>
        >& output
    );

    void collect_named_parameters(
        const std::string& prefix,
        std::vector<
            std::pair<std::string, const Parameter*>
        >& output
    ) const;
};

} // namespace minitorch
