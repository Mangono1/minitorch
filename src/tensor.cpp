#include "minitorch/tensor.h"

#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace minitorch {

Tensor::Tensor()
    : data_(),
      shape_(),
      device_(DeviceType::CPU),
      dtype_(DType::Float32) {}

Tensor::Tensor(
    const std::vector<float>& data,
    const std::vector<std::size_t>& shape,
    Device device
)
    : data_(data),
      shape_(shape),
      device_(device),
      dtype_(DType::Float32) {

    validate();
}

Tensor::Tensor(
    std::initializer_list<float> data,
    std::initializer_list<std::size_t> shape,
    Device device
)
    : data_(data),
      shape_(shape),
      device_(device),
      dtype_(DType::Float32) {

    validate();
}

const std::vector<float>& Tensor::data() const {
    return data_;
}

std::vector<float>& Tensor::data() {
    return data_;
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

std::size_t Tensor::ndim() const {
    return shape_.size();
}

std::size_t Tensor::size() const {
    return data_.size();
}

Device Tensor::device() const {
    return device_;
}

DType Tensor::dtype() const {
    return dtype_;
}

bool Tensor::empty() const {
    return data_.empty();
}

float Tensor::item(std::size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor item index out of range");
    }

    return data_[index];
}

void Tensor::set_item(std::size_t index, float value) {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor item index out of range");
    }

    data_[index] = value;
}

void Tensor::validate() const {
    if (shape_.empty()) {
        if (!data_.empty()) {
            throw std::invalid_argument(
                "Scalar tensor cannot contain multiple elements"
            );
        }

        return;
    }

    std::size_t expected_size = 1;

    for (std::size_t dimension : shape_) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions cannot be zero"
            );
        }

        expected_size *= dimension;
    }

    if (expected_size != data_.size()) {
        throw std::invalid_argument(
            "Tensor data size does not match tensor shape"
        );
    }
}

std::string Tensor::repr() const {
    std::ostringstream out;

    out << "Tensor(";
    out << "shape=[";

    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }

        out << shape_[i];
    }

    out << "], ";
    out << "dtype=float32, ";
    out << "device=" << device_.name();
    out << ", data=[";

    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }

        out << std::fixed << std::setprecision(4) << data_[i];
    }

    out << "])";

    return out.str();
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Tensor add requires identical shapes"
        );
    }

    if (device_.type() != other.device_.type()) {
        throw std::invalid_argument(
            "Tensor add requires tensors on the same device"
        );
    }

    std::vector<float> result(data_.size());

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] + other.data_[i];
    }

    return Tensor(result, shape_, device_);
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Tensor subtract requires identical shapes"
        );
    }

    if (device_.type() != other.device_.type()) {
        throw std::invalid_argument(
            "Tensor subtract requires tensors on the same device"
        );
    }

    std::vector<float> result(data_.size());

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] - other.data_[i];
    }

    return Tensor(result, shape_, device_);
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Tensor multiply requires identical shapes"
        );
    }

    if (device_.type() != other.device_.type()) {
        throw std::invalid_argument(
            "Tensor multiply requires tensors on the same device"
        );
    }

    std::vector<float> result(data_.size());

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] * other.data_[i];
    }

    return Tensor(result, shape_, device_);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument(
            "matmul currently requires two 2D tensors"
        );
    }

    if (device_.type() != other.device_.type()) {
        throw std::invalid_argument(
            "matmul requires tensors on the same device"
        );
    }

    const std::size_t rows_a = shape_[0];
    const std::size_t cols_a = shape_[1];

    const std::size_t rows_b = other.shape_[0];
    const std::size_t cols_b = other.shape_[1];

    if (cols_a != rows_b) {
        throw std::invalid_argument(
            "matmul dimension mismatch"
        );
    }

    std::vector<float> result(rows_a * cols_b, 0.0f);

    for (std::size_t i = 0; i < rows_a; ++i) {
        for (std::size_t k = 0; k < cols_a; ++k) {
            const float a = data_[i * cols_a + k];

            for (std::size_t j = 0; j < cols_b; ++j) {
                result[i * cols_b + j] +=
                    a * other.data_[k * cols_b + j];
            }
        }
    }

    return Tensor(
        result,
        {rows_a, cols_b},
        device_
    );
}

} // namespace minitorch
