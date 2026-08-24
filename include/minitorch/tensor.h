#pragma once

#include "minitorch/device.h"
#include "minitorch/dtype.h"

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace minitorch {

class Tensor {
public:
    Tensor();
    Tensor(
        const std::vector<float>& data,
        const std::vector<std::size_t>& shape,
        Device device = Device(DeviceType::CPU)
    );

    Tensor(
        std::initializer_list<float> data,
        std::initializer_list<std::size_t> shape,
        Device device = Device(DeviceType::CPU)
    );

    const std::vector<float>& data() const;
    std::vector<float>& data();

    const std::vector<std::size_t>& shape() const;

    std::size_t ndim() const;
    std::size_t size() const;

    Device device() const;
    DType dtype() const;

    bool empty() const;

    float item(std::size_t index) const;
    void set_item(std::size_t index, float value);

    std::string repr() const;

    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    Device device_;
    DType dtype_;

    void validate() const;
};

} // namespace minitorch
