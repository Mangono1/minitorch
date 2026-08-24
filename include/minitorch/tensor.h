#pragma once

#include "minitorch/device.h"
#include "minitorch/dtype.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace minitorch {

struct TensorImpl;

class Tensor {
public:
    Tensor();

    Tensor(
        const std::vector<float>& data,
        const std::vector<std::size_t>& shape,
        Device device = Device(DeviceType::CPU),
        bool requires_grad = false
    );

    Tensor(
        std::initializer_list<float> data,
        std::initializer_list<std::size_t> shape,
        Device device = Device(DeviceType::CPU),
        bool requires_grad = false
    );

    const std::vector<float>& data() const;
    std::vector<float>& data();

    const std::vector<std::size_t>& shape() const;

    std::size_t ndim() const;
    std::size_t size() const;

    Device device() const;
    DType dtype() const;

    bool empty() const;

    bool requires_grad() const;
    bool has_grad() const;

    Tensor grad() const;

    void zero_grad();
    void backward();

    float item(std::size_t index) const;
    void set_item(std::size_t index, float value);

    std::string repr() const;

    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;
    Tensor multiply_scalar(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;
    Tensor relu() const;
    Tensor add_bias_2d(const Tensor& bias) const;

private:
    std::shared_ptr<TensorImpl> impl_;

    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    void validate() const;

    friend struct TensorImpl;
};

} // namespace minitorch
