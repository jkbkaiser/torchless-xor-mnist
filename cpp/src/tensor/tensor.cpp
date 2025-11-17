#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

#include <torchless/tensor.h>

//
// Helpers
//

void same_device_check(Tensor a, Tensor b) { assert(a.device_ == b.device_); }

//
// Factories
//

Tensor::Tensor(const std::vector<float> &values, Device device)
    : device_(device), tensor_(tensor_from_data(values, device)) {}

Tensor::Tensor(const std::vector<std::vector<float>> &values, Device device)
    : device_(device), tensor_(tensor_from_data(values, device)) {}

Tensor::Tensor(const Shape &shape, float value, Device device)
    : device_(device), tensor_(filled_tensor(shape, value, device)) {}

Tensor::Tensor(const Shape &shape, std::mt19937 rng, Device device)
    : device_(device), tensor_(random_tensor(shape, rng, device)) {}

Tensor Tensor::empty(const Shape &shape, Device device) {
    return Tensor(shape, device, EmptyTag{});
}

Tensor Tensor::filled(const Shape &shape, float value, Device device) {
    return Tensor(shape, value, device);
}

Tensor Tensor::zeros(const Shape &shape, Device device) { return Tensor::filled(shape, 0, device); }

Tensor Tensor::ones(const Shape &shape, Device device) { return Tensor::filled(shape, 1, device); }

Tensor Tensor::rand(const Shape &shape, std::mt19937 rng, Device device) {
    return Tensor(shape, rng, device);
}

Tensor::Tensor(const Shape &shape, Device device, EmptyTag)
    : device_(device), tensor_(tensor_from_shape(shape, device)) {}

Tensor::Tensor(TensorVariant &tensor, Device device) : device_(device), tensor_(tensor) {}

TensorVariant Tensor::tensor_from_shape(const Shape &shape, Device device) {
    switch (device) {
    case Device::CPU:
        return CPUTensor(shape);
    case Device::GPU:
        return GPUTensor(shape);
    default:
        throw std::runtime_error("tensor_from_shape: Unknown device");
    }
}

TensorVariant Tensor::tensor_from_data(const std::vector<float> &values, Device device) {
    switch (device) {
    case Device::CPU:
        return CPUTensor(values);
    case Device::GPU:
        return GPUTensor(values);
    default:
        throw std::runtime_error("tensor_from_data_1d: Unknown device");
    }
}

TensorVariant Tensor::tensor_from_data(const std::vector<std::vector<float>> &values,
                                       Device device) {
    switch (device) {
    case Device::CPU:
        return CPUTensor(values);
    case Device::GPU:
        return GPUTensor(values);
    default:
        throw std::runtime_error("tensor_from_data_2d: Unknown device");
    }
}

TensorVariant Tensor::filled_tensor(const Shape &shape, float value, Device device) {
    switch (device) {
    case Device::CPU:
        return CPUTensor::filled(shape, value);
    case Device::GPU:
        return GPUTensor::filled(shape, value);
    default:
        throw std::runtime_error("filled_tensor: Unknown device");
    }
}

TensorVariant Tensor::random_tensor(const Shape &shape, std::mt19937 rng, Device device) {
    switch (device) {
    case Device::CPU:
        return CPUTensor::rand(shape, rng);
    case Device::GPU:
        return GPUTensor::rand(shape, rng);
    default:
        throw std::runtime_error("random_tensor: Unknown device");
    }
}

//
// Operators
//

Tensor Tensor::log() const {
    TensorVariant new_tensor =
        std::visit([&](auto &tensor_impl) -> TensorVariant { return tensor_impl.log(); }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::exp() const {
    TensorVariant new_tensor =
        std::visit([&](auto &tensor_impl) -> TensorVariant { return tensor_impl.exp(); }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator-() const {
    TensorVariant new_tensor =
        std::visit([&](auto &tensor_impl) -> TensorVariant { return -tensor_impl; }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator+(float scalar) const {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl + scalar; }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator-(float scalar) const {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl - scalar; }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator*(float scalar) const {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl * scalar; }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator/(float scalar) const {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl / scalar; }, tensor_);
    return Tensor(new_tensor, device_);
}

Tensor Tensor::operator==(const Tensor &other) const {
    same_device_check(*this, other);

    switch (device_) {
    case Device::CPU: {
        const CPUTensor &a = std::get<CPUTensor>(tensor_);
        const CPUTensor &b = std::get<CPUTensor>(other.tensor_);
        TensorVariant result_variant = a == b;
        return Tensor(result_variant, device_);
    }
    case Device::GPU: {
        const GPUTensor &a = std::get<GPUTensor>(tensor_);
        const GPUTensor &b = std::get<GPUTensor>(other.tensor_);
        TensorVariant result_variant = a == b;
        return Tensor(result_variant, device_);
    }
    default:
        throw std::runtime_error("operator==: Unknown device");
    }
}

float Tensor::get(const std::vector<size_t> &indices) {
    return std::visit([&](auto &tensor_impl) -> float { return tensor_impl.get(indices); },
                      tensor_);
}

Tensor operator+(float scalar, const Tensor &tensor) {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl + scalar; }, tensor.tensor_);
    return Tensor(new_tensor, tensor.device_);
}

Tensor operator-(float scalar, const Tensor &tensor) {
    TensorVariant new_tensor =
        std::visit([&](auto &tensor_impl) -> TensorVariant { return (-tensor_impl) + scalar; },
                   tensor.tensor_);
    return Tensor(new_tensor, tensor.device_);
}

Tensor operator*(float scalar, const Tensor &tensor) {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return tensor_impl * scalar; }, tensor.tensor_);
    return Tensor(new_tensor, tensor.device_);
}

Tensor operator/(float scalar, const Tensor &tensor) {
    TensorVariant new_tensor = std::visit(
        [&](auto &tensor_impl) -> TensorVariant { return scalar / tensor_impl; }, tensor.tensor_);
    return Tensor(new_tensor, tensor.device_);
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    os << "Tensor(\n\tDevice: ";
    switch (t.device_) {
    case Device::CPU:
        os << "CPU,\n\tData: ";
        std::visit([&os](const auto &tensor_impl) { os << tensor_impl; }, t.tensor_);
        os << ")";
        return os;
    case Device::GPU:
        os << "GPU,\n\tData: ";
        std::visit([&os](const auto &tensor_impl) { os << tensor_impl; }, t.tensor_);
        os << ")";
        return os;
    default:
        throw std::runtime_error("&operator: Unknown device");
    }
}

Tensor Tensor::to(Device dev) const {
    if (dev == device_)
        return *this;

    switch (dev) {
    case Device::CPU: {
        GPUTensor gpu_tensor = std::get<GPUTensor>(tensor_);
        TensorVariant m_cpu_tensor = gpu_tensor.toCPU();
        return Tensor(m_cpu_tensor, Device::CPU);
    }
    case Device::GPU: {
        CPUTensor cpu_tensor = std::get<CPUTensor>(tensor_);
        TensorVariant m_gpu_tensor = cpu_tensor.toGPU();
        return Tensor(m_gpu_tensor, Device::GPU);
    }
    default:
        throw std::runtime_error("to: Unknown device");
    }
}

// Lets use a template for this!!
// #include <algorithm>
// #include <functional>
// #include <iomanip>
// #include <iostream>
// #include <numeric>
// #include <random>
// #include <vector>

// #include <torchless/tensor.h>

// Tensor Tensor::squeeze() const {
//     std::vector<size_t> new_shape;
//     for (size_t dim : shape) {
//         if (dim != 1) {
//             new_shape.push_back(dim);
//         }
//     }
//
//     if (new_shape.empty()) {
//         new_shape.push_back(1);
//     }
//
//     Tensor result(new_shape);
//     result.data = data;
//     return result;
// }

// std::ostream &operator<<(std::ostream &os, const Tensor &t) {
//     int rows = 10;
//     int dotted_rows = 3;
//     os << std::fixed << std::setprecision(3);
//     print_recursive(os, t.data, t.shape, 0, 0, rows, dotted_rows);
//     return os;
// }
//
// Tensor Tensor::transpose() const {
//     if (this->shape.size() != 2) {
//         throw std::runtime_error("Transpose is only supported for 2D matrices");
//     }
//
//     Tensor t = Tensor({this->shape[1], this->shape[0]});
//
//     for (size_t i = 0; i < this->shape[0]; ++i) {
//         for (size_t j = 0; j < this->shape[1]; ++j) {
//             t.at({j, i}) = this->at({i, j});
//         }
//     }
//
//     return t;
// }
//
// Tensor Tensor::add_dim(const int dim) const {
//     Tensor result(this->shape);
//     result.data = this->data;
//     result.shape.insert(result.shape.begin() + dim, 1);
//     return result;
// }
//
// Tensor Tensor::map(const std::function<float(float)> &func) const {
//     Tensor result(this->shape);
//     std::transform(this->data.begin(), this->data.end(), result.data.begin(), func);
//     return result;
// }
//
// float &Tensor::at(const std::vector<size_t> &indices) {
//     int offset = 0;
//     int stride = 1;
//     for (int i = (int)shape.size() - 1; i >= 0; --i) {
//         offset += indices[i] * stride;
//         stride *= shape[i];
//     }
//     return data[offset];
// }
//
// Tensor Tensor::max() const {
//     float max = data[0];
//     for (size_t i = 1; i < data.size(); ++i) {
//         if (data[i] > max) {
//             max = data[i];
//         }
//     }
//     return Tensor::filled({1}, max);
// }
//
// Tensor Tensor::sum() const {
//     float sum = 0.0;
//     for (size_t i = 0; i < data.size(); ++i) {
//         sum += data[i];
//     }
//     return Tensor::filled({1}, sum);
// }
//
// Tensor Tensor::sum(int axis) const {
//     if (axis < 0)
//         axis = shape.size() + axis;
//     Tensor t = Tensor({shape[1]});
//
//     if (axis == 1) {
//         for (size_t i = 0; i < shape[0]; ++i) {
//             for (size_t j = 0; j < shape[1]; ++j) {
//                 t.at({j}) += this->at({i, j});
//             }
//         }
//     } else {
//         throw std::runtime_error("Sum only supported for axis 1");
//     }
//
//     return t;
// }
//
// float Tensor::item() const { return data[0]; }
//
// float Tensor::at(const std::vector<size_t> &indices) const {
//     int offset = 0;
//     int stride = 1;
//     for (int i = (int)shape.size() - 1; i >= 0; --i) {
//         offset += indices[i] * stride;
//         stride *= shape[i];
//     }
//     return data[offset];
// }
//
// Tensor Tensor::dot(Tensor other) const {
//     if (this->shape.size() != 1 || other.shape.size() != 1) {
//         throw std::runtime_error("Dot product is only defined for 1D tensors.");
//     }
//     if (this->shape[0] != other.shape[0]) {
//         throw std::runtime_error(
//             "Dot product is only defined for tensors with compatible dimensions.");
//     }
//
//     float result = 0.0;
//     for (size_t i = 0; i < this->shape[0]; ++i) {
//         result += this->at({i}) * other.at({i});
//     }
//
//     return Tensor::filled({1}, result);
// }
//
// Tensor Tensor::matmul(Tensor other) const {
//     // Dot product
//     if (this->shape.size() == 1 && other.shape.size() == 1) {
//         return this->dot(other);
//     }
//
//     // Matrix-matrix multiplication
//     if (this->shape.size() == 2 && other.shape.size() == 2) {
//         if (this->shape[1] != other.shape[0]) {
//             throw std::runtime_error(
//                 "Matrix-matrix multiplication is only defined for compatible dimensions.");
//         }
//
//         auto result = Tensor({this->shape[0], other.shape[1]});
//
//         for (size_t i = 0; i < this->shape[0]; ++i) {
//             for (size_t j = 0; j < other.shape[1]; ++j) {
//                 float sum = 0.0;
//                 for (size_t k = 0; k < this->shape[1]; ++k) {
//                     sum += this->at({i, k}) * other.at({k, j});
//                 }
//                 result.at({i, j}) = sum;
//             }
//         }
//
//         return result;
//     }
//
//     // Matrix-vector multiplication
//     if (this->shape.size() == 2 && other.shape.size() == 1) {
//         if (this->shape[1] != other.shape[0]) {
//             throw std::runtime_error(
//                 "Matrix-vector multiplication is only defined for compatible dimensions.");
//         }
//
//         auto result = Tensor({this->shape[0]});
//
//         for (size_t i = 0; i < this->shape[0]; ++i) {
//             float sum = 0.0;
//             for (size_t k = 0; k < this->shape[1]; ++k) {
//                 sum += this->at({i, k}) * other.at({k});
//             }
//             result.at({i}) = sum;
//         }
//
//         return result;
//     }
//
//     // Matrix-vector multiplication
//     if (this->shape.size() == 1 && other.shape.size() == 2) {
//         if (this->shape[0] != other.shape[0]) {
//             throw std::runtime_error(
//                 "Matrix-vector multiplication is only defined for compatible dimensions.");
//         }
//
//         auto result = Tensor({other.shape[1]});
//
//         for (size_t j = 0; j < other.shape[1]; ++j) {
//             float sum = 0.0;
//             for (size_t k = 0; k < this->shape[0]; ++k) {
//                 sum += this->at({k}) * other.at({k, j});
//             }
//             result.at({j}) = sum;
//         }
//
//         return result;
//     }
//
//     throw std::runtime_error("Matrix multiplication is not yet implemented for this input.");
// }
//
// Tensor stack(const std::vector<Tensor> &tensors, int axis) {
//     if (tensors.size() == 0) {
//         throw std::runtime_error("Cannot stack an empty list of tensors.");
//     }
//
//     if (tensors.size() == 1) {
//         return tensors[0];
//     }
//
//     for (size_t i = 0; i < tensors.size(); ++i) {
//         if (tensors[i].shape != tensors[0].shape) {
//             throw std::runtime_error("Tensors must have the same shape to be stacked.");
//         }
//     }
//
//     const int d = tensors[0].shape.size();
//
//     if (d == 1) {
//         if (axis == 0) {
//             std::vector<size_t> shape = {tensors.size(), tensors[0].shape[0]};
//
//             Tensor result = Tensor(shape);
//
//             for (size_t i = 0; i < tensors[0].shape[0]; ++i) {
//                 for (size_t j = 0; j < tensors.size(); ++j) {
//                     result.at({j, i}) = tensors[j].at({i});
//                 }
//             }
//             return result;
//         } else if (axis == 1) {
//             std::vector<size_t> shape = {tensors[0].shape[0], tensors.size()};
//
//             Tensor result = Tensor(shape);
//
//             for (size_t i = 0; i < tensors[0].shape[0]; ++i) {
//                 for (size_t j = 0; j < tensors.size(); ++j) {
//                     result.at({i, j}) = tensors[j].at({i});
//                 }
//             }
//             return result;
//         } else {
//             throw std::runtime_error("Axis must be 0 or 1.");
//         }
//     }
//
//     throw std::runtime_error("Stacking is only supported for tensors with one dimensions.");
// }
