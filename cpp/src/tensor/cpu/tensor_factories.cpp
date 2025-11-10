// Tensor factory functions - create new tensors
// Creation from scratch:
//   - zeros(shape)         - All zeros
//   - ones(shape)          - All ones
//   - filled(shape, value) - All same value
//   - rand(shape)          - Random values [0, 1)
//
// Creation from data:
//   - from_vec(vector<float>)
//   - from_vec(vector<float>, shape)
//   - from_vec(vector<vector<float>>)

#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <torchless/shape.h>
#include <torchless/tensor/tensor_base.h>
#include <torchless/tensor/tensor_cpu.h>

CPUTensor::CPUTensor(const Shape &shape) : BaseTensor(shape) {
    size_t size = shape.numel();
    data_.resize(size);
}

CPUTensor::CPUTensor(const std::vector<float> &values) : BaseTensor({values.size()}) {
    data_.resize(values.size());
    data_ = values;
}

CPUTensor::CPUTensor(const std::vector<std::vector<float>> &values)
    : BaseTensor({values.size(), values[0].size()}) {
    size_t rows = values.size();
    size_t cols = values[0].size();
    data_.resize(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
        const auto &row = values[i];

        for (size_t j = 0; j < cols; ++j) {
            data_[i * cols + j] = row[j];
        }
    }
}

CPUTensor CPUTensor::filled(const Shape &shape, float value) {
    CPUTensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), value);
    return t;
}

CPUTensor CPUTensor::rand(const Shape &shape, std::mt19937 &rng) {
    CPUTensor t(shape);
    std::uniform_real_distribution<float> d(0.0, 1.0);
    std::generate(t.data_.begin(), t.data_.end(), std::bind(d, rng));
    return t;
}

float CPUTensor::get(const std::vector<size_t> &indices) {
    int offset = 0;
    int stride = 1;
    for (int i = (int)shape_.size() - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape_[i];
    }
    return data_[offset];
}
