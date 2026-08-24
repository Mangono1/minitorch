#include "minitorch/tensor.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minitorch {

struct AutogradNode {
    std::vector<std::shared_ptr<TensorImpl>> parents;

    std::function<void(const std::vector<float>&)> backward_fn;
};

struct TensorImpl {
    std::vector<float> data;
    std::vector<std::size_t> shape;

    Device device;
    DType dtype;

    bool requires_grad = false;

    std::vector<float> grad;

    std::shared_ptr<AutogradNode> grad_fn;
};

static void ensure_cpu(const TensorImpl& tensor) {
    if (!tensor.device.is_cpu()) {
        throw std::runtime_error(
            "This operation is not implemented for Vulkan yet"
        );
    }
}

static void ensure_same_device(
    const TensorImpl& a,
    const TensorImpl& b
) {
    if (a.device.type() != b.device.type()) {
        throw std::invalid_argument(
            "Tensor operation requires tensors on the same device"
        );
    }
}

static void accumulate_gradient(
    const std::shared_ptr<TensorImpl>& tensor,
    const std::vector<float>& gradient
) {
    if (!tensor->requires_grad) {
        return;
    }

    if (tensor->grad.empty()) {
        tensor->grad.assign(
            gradient.size(),
            0.0f
        );
    }

    if (tensor->grad.size() != gradient.size()) {
        throw std::runtime_error(
            "Gradient size mismatch"
        );
    }

    for (std::size_t i = 0; i < gradient.size(); ++i) {
        tensor->grad[i] += gradient[i];
    }
}

Tensor::Tensor()
    : impl_(
        std::make_shared<TensorImpl>()
    ) {

    impl_->device = Device(DeviceType::CPU);
    impl_->dtype = DType::Float32;
}

Tensor::Tensor(
    const std::vector<float>& data,
    const std::vector<std::size_t>& shape,
    Device device,
    bool requires_grad
)
    : impl_(
        std::make_shared<TensorImpl>()
    ) {

    impl_->data = data;
    impl_->shape = shape;
    impl_->device = device;
    impl_->dtype = DType::Float32;
    impl_->requires_grad = requires_grad;

    validate();
}

Tensor::Tensor(
    std::initializer_list<float> data,
    std::initializer_list<std::size_t> shape,
    Device device,
    bool requires_grad
)
    : impl_(
        std::make_shared<TensorImpl>()
    ) {

    impl_->data = data;
    impl_->shape = shape;
    impl_->device = device;
    impl_->dtype = DType::Float32;
    impl_->requires_grad = requires_grad;

    validate();
}

Tensor::Tensor(
    std::shared_ptr<TensorImpl> impl
)
    : impl_(std::move(impl)) {}

const std::vector<float>& Tensor::data() const {
    return impl_->data;
}

std::vector<float>& Tensor::data() {
    return impl_->data;
}

const std::vector<std::size_t>& Tensor::shape() const {
    return impl_->shape;
}

std::size_t Tensor::ndim() const {
    return impl_->shape.size();
}

std::size_t Tensor::size() const {
    return impl_->data.size();
}

Device Tensor::device() const {
    return impl_->device;
}

DType Tensor::dtype() const {
    return impl_->dtype;
}

bool Tensor::empty() const {
    return impl_->data.empty();
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad;
}

bool Tensor::has_grad() const {
    return !impl_->grad.empty();
}

Tensor Tensor::grad() const {
    if (!impl_->requires_grad) {
        throw std::runtime_error(
            "Tensor does not require gradients"
        );
    }

    if (impl_->grad.empty()) {
        return Tensor(
            std::vector<float>(
                impl_->data.size(),
                0.0f
            ),
            impl_->shape,
            impl_->device,
            false
        );
    }

    return Tensor(
        impl_->grad,
        impl_->shape,
        impl_->device,
        false
    );
}

void Tensor::zero_grad() {
    if (!impl_->requires_grad) {
        return;
    }

    impl_->grad.assign(
        impl_->data.size(),
        0.0f
    );
}

float Tensor::item(std::size_t index) const {
    if (index >= impl_->data.size()) {
        throw std::out_of_range(
            "Tensor item index out of range"
        );
    }

    return impl_->data[index];
}

void Tensor::set_item(
    std::size_t index,
    float value
) {
    if (index >= impl_->data.size()) {
        throw std::out_of_range(
            "Tensor item index out of range"
        );
    }

    impl_->data[index] = value;
}

void Tensor::validate() const {
    if (impl_->shape.empty()) {
        if (!impl_->data.empty()) {
            throw std::invalid_argument(
                "Scalar tensor cannot contain multiple elements"
            );
        }

        return;
    }

    std::size_t expected_size = 1;

    for (std::size_t dimension : impl_->shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions cannot be zero"
            );
        }

        expected_size *= dimension;
    }

    if (expected_size != impl_->data.size()) {
        throw std::invalid_argument(
            "Tensor data size does not match tensor shape"
        );
    }
}

std::string Tensor::repr() const {
    std::ostringstream out;

    out << "Tensor(";

    out << "shape=[";

    for (std::size_t i = 0; i < impl_->shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }

        out << impl_->shape[i];
    }

    out << "], ";

    out << "dtype=float32, ";

    out << "device=" << impl_->device.name();

    out << ", requires_grad="
        << (impl_->requires_grad ? "true" : "false");

    out << ", data=[";

    for (std::size_t i = 0; i < impl_->data.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }

        out << std::fixed
            << std::setprecision(4)
            << impl_->data[i];
    }

    out << "])";

    return out.str();
}

Tensor Tensor::add(
    const Tensor& other
) const {

    ensure_cpu(*impl_);
    ensure_cpu(*other.impl_);

    ensure_same_device(
        *impl_,
        *other.impl_
    );

    if (impl_->shape != other.impl_->shape) {
        throw std::invalid_argument(
            "Tensor add requires identical shapes"
        );
    }

    std::vector<float> result(
        impl_->data.size()
    );

    for (std::size_t i = 0;
         i < result.size();
         ++i) {

        result[i] =
            impl_->data[i] +
            other.impl_->data[i];
    }

    const bool needs_grad =
        impl_->requires_grad ||
        other.impl_->requires_grad;

    auto output = std::make_shared<TensorImpl>();

    output->data = result;
    output->shape = impl_->shape;
    output->device = impl_->device;
    output->dtype = DType::Float32;
    output->requires_grad = needs_grad;

    if (needs_grad) {
        auto node =
            std::make_shared<AutogradNode>();

        node->parents = {
            impl_,
            other.impl_
        };

        node->backward_fn =
            [a = impl_, b = other.impl_](
                const std::vector<float>& grad
            ) {

                accumulate_gradient(a, grad);
                accumulate_gradient(b, grad);
            };

        output->grad_fn = node;
    }

    return Tensor(output);
}

Tensor Tensor::subtract(
    const Tensor& other
) const {

    ensure_cpu(*impl_);
    ensure_cpu(*other.impl_);

    ensure_same_device(
        *impl_,
        *other.impl_
    );

    if (impl_->shape != other.impl_->shape) {
        throw std::invalid_argument(
            "Tensor subtract requires identical shapes"
        );
    }

    std::vector<float> result(
        impl_->data.size()
    );

    for (std::size_t i = 0;
         i < result.size();
         ++i) {

        result[i] =
            impl_->data[i] -
            other.impl_->data[i];
    }

    const bool needs_grad =
        impl_->requires_grad ||
        other.impl_->requires_grad;

    auto output = std::make_shared<TensorImpl>();

    output->data = result;
    output->shape = impl_->shape;
    output->device = impl_->device;
    output->dtype = DType::Float32;
    output->requires_grad = needs_grad;

    if (needs_grad) {
        auto node =
            std::make_shared<AutogradNode>();

        node->parents = {
            impl_,
            other.impl_
        };

        node->backward_fn =
            [a = impl_, b = other.impl_](
                const std::vector<float>& grad
            ) {

                accumulate_gradient(a, grad);

                std::vector<float> negative(
                    grad.size()
                );

                for (std::size_t i = 0;
                     i < grad.size();
                     ++i) {

                    negative[i] =
                        -grad[i];
                }

                accumulate_gradient(
                    b,
                    negative
                );
            };

        output->grad_fn = node;
    }

    return Tensor(output);
}

Tensor Tensor::multiply(
    const Tensor& other
) const {

    ensure_cpu(*impl_);
    ensure_cpu(*other.impl_);

    ensure_same_device(
        *impl_,
        *other.impl_
    );

    if (impl_->shape != other.impl_->shape) {
        throw std::invalid_argument(
            "Tensor multiply requires identical shapes"
        );
    }

    std::vector<float> result(
        impl_->data.size()
    );

    for (std::size_t i = 0;
         i < result.size();
         ++i) {

        result[i] =
            impl_->data[i] *
            other.impl_->data[i];
    }

    const bool needs_grad =
        impl_->requires_grad ||
        other.impl_->requires_grad;

    auto output = std::make_shared<TensorImpl>();

    output->data = result;
    output->shape = impl_->shape;
    output->device = impl_->device;
    output->dtype = DType::Float32;
    output->requires_grad = needs_grad;

    if (needs_grad) {
        auto node =
            std::make_shared<AutogradNode>();

        node->parents = {
            impl_,
            other.impl_
        };

        node->backward_fn =
            [a = impl_, b = other.impl_](
                const std::vector<float>& grad
            ) {

                if (a->requires_grad) {
                    std::vector<float> ga(
                        grad.size()
                    );

                    for (std::size_t i = 0;
                         i < grad.size();
                         ++i) {

                        ga[i] =
                            grad[i] *
                            b->data[i];
                    }

                    accumulate_gradient(
                        a,
                        ga
                    );
                }

                if (b->requires_grad) {
                    std::vector<float> gb(
                        grad.size()
                    );

                    for (std::size_t i = 0;
                         i < grad.size();
                         ++i) {

                        gb[i] =
                            grad[i] *
                            a->data[i];
                    }

                    accumulate_gradient(
                        b,
                        gb
                    );
                }
            };

        output->grad_fn = node;
    }

    return Tensor(output);
}

Tensor Tensor::matmul(
    const Tensor& other
) const {

    ensure_cpu(*impl_);
    ensure_cpu(*other.impl_);

    ensure_same_device(
        *impl_,
        *other.impl_
    );

    if (
        impl_->shape.size() != 2 ||
        other.impl_->shape.size() != 2
    ) {
        throw std::invalid_argument(
            "matmul currently requires two 2D tensors"
        );
    }

    const std::size_t rows_a =
        impl_->shape[0];

    const std::size_t cols_a =
        impl_->shape[1];

    const std::size_t rows_b =
        other.impl_->shape[0];

    const std::size_t cols_b =
        other.impl_->shape[1];

    if (cols_a != rows_b) {
        throw std::invalid_argument(
            "matmul dimension mismatch"
        );
    }

    std::vector<float> result(
        rows_a * cols_b,
        0.0f
    );

    for (std::size_t i = 0;
         i < rows_a;
         ++i) {

        for (std::size_t k = 0;
             k < cols_a;
             ++k) {

            const float a =
                impl_->data[i * cols_a + k];

            for (std::size_t j = 0;
                 j < cols_b;
                 ++j) {

                result[
                    i * cols_b + j
                ] +=
                    a *
                    other.impl_->data[
                        k * cols_b + j
                    ];
            }
        }
    }

    const bool needs_grad =
        impl_->requires_grad ||
        other.impl_->requires_grad;

    auto output =
        std::make_shared<TensorImpl>();

    output->data = result;

    output->shape = {
        rows_a,
        cols_b
    };

    output->device = impl_->device;
    output->dtype = DType::Float32;
    output->requires_grad = needs_grad;

    if (needs_grad) {
        auto node =
            std::make_shared<AutogradNode>();

        node->parents = {
            impl_,
            other.impl_
        };

        node->backward_fn =
            [
                a = impl_,
                b = other.impl_,
                rows_a,
                cols_a,
                cols_b
            ](
                const std::vector<float>& grad
            ) {

                if (a->requires_grad) {
                    std::vector<float> ga(
                        rows_a * cols_a,
                        0.0f
                    );

                    for (std::size_t i = 0;
                         i < rows_a;
                         ++i) {

                        for (std::size_t k = 0;
                             k < cols_a;
                             ++k) {

                            float value = 0.0f;

                            for (std::size_t j = 0;
                                 j < cols_b;
                                 ++j) {

                                value +=
                                    grad[
                                        i * cols_b + j
                                    ] *
                                    b->data[
                                        k * cols_b + j
                                    ];
                            }

                            ga[
                                i * cols_a + k
                            ] += value;
                        }
                    }

                    accumulate_gradient(
                        a,
                        ga
                    );
                }

                if (b->requires_grad) {
                    std::vector<float> gb(
                        cols_a * cols_b,
                        0.0f
                    );

                    for (std::size_t k = 0;
                         k < cols_a;
                         ++k) {

                        for (std::size_t j = 0;
                             j < cols_b;
                             ++j) {

                            float value = 0.0f;

                            for (std::size_t i = 0;
                                 i < rows_a;
                                 ++i) {

                                value +=
                                    a->data[
                                        i * cols_a + k
                                    ] *
                                    grad[
                                        i * cols_b + j
                                    ];
                            }

                            gb[
                                k * cols_b + j
                            ] += value;
                        }
                    }

                    accumulate_gradient(
                        b,
                        gb
                    );
                }
            };

        output->grad_fn = node;
    }

    return Tensor(output);
}

Tensor Tensor::sum() const {
    ensure_cpu(*impl_);

    float total = 0.0f;

    for (float value : impl_->data) {
        total += value;
    }

    auto output =
        std::make_shared<TensorImpl>();

    output->data = {total};
    output->shape = {1};
    output->device = impl_->device;
    output->dtype = DType::Float32;
    output->requires_grad =
        impl_->requires_grad;

    if (impl_->requires_grad) {
        auto node =
            std::make_shared<AutogradNode>();

        node->parents = {
            impl_
        };

        node->backward_fn =
            [a = impl_](
                const std::vector<float>& grad
            ) {

                if (grad.size() != 1) {
                    throw std::runtime_error(
                        "sum backward expects scalar gradient"
                    );
                }

                std::vector<float> ga(
                    a->data.size(),
                    grad[0]
                );

                accumulate_gradient(
                    a,
                    ga
                );
            };

        output->grad_fn = node;
    }

    return Tensor(output);
}

void Tensor::backward() {
    if (!impl_->requires_grad) {
        throw std::runtime_error(
            "Cannot call backward() on a tensor "
            "that does not require gradients"
        );
    }

    if (impl_->data.size() != 1) {
        throw std::runtime_error(
            "backward() currently requires a scalar tensor"
        );
    }

    impl_->grad.assign(
        1,
        1.0f
    );

    std::vector<
        std::shared_ptr<TensorImpl>
    > topology;

    std::set<TensorImpl*> visited;

    std::function<void(
        const std::shared_ptr<TensorImpl>&
    )> build_topology;

    build_topology =
        [&](const std::shared_ptr<TensorImpl>& node) {

            if (!node) {
                return;
            }

            if (visited.count(node.get())) {
                return;
            }

            visited.insert(node.get());

            if (node->grad_fn) {
                for (
                    const auto& parent :
                    node->grad_fn->parents
                ) {
                    build_topology(parent);
                }
            }

            topology.push_back(node);
        };

    build_topology(impl_);

    for (
        auto it = topology.rbegin();
        it != topology.rend();
        ++it
    ) {
        const auto& node = *it;

        if (
            node->grad_fn &&
            !node->grad.empty()
        ) {
            node->grad_fn->backward_fn(
                node->grad
            );
        }
    }
}

} // namespace minitorch
