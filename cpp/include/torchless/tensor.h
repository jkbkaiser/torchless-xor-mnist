#ifndef TORCHLESS_TENSOR_H
#define TORCHLESS_TENSOR_H

#include <iostream>
#include <ostream>
#include <random>
#include <variant>
#include <vector>

#include <torchless/shape.h>
#include <torchless/tensor/tensor_base.h>
#include <torchless/tensor/tensor_cpu.h>
#include <torchless/tensor/tensor_gpu.h>

enum class Device {
    CPU,
    GPU,
};

struct EmptyTag {};

using TensorVariant = std::variant<CPUTensor, GPUTensor>;

class Tensor {
  public:
    Device device_;
    TensorVariant tensor_;

    // Factories
    Tensor(const std::vector<float> &values, Device device = Device::CPU);
    Tensor(const std::vector<std::vector<float>> &values, Device device = Device::CPU);
    static Tensor empty(const Shape &shape, Device device = Device::CPU);
    static Tensor zeros(const Shape &shape, Device device = Device::CPU);
    static Tensor ones(const Shape &shape, Device device = Device::CPU);
    static Tensor filled(const Shape &shape, float value, Device device = Device::CPU);
    static Tensor rand(const Shape &shape, std::mt19937 rng, Device device = Device::CPU);

    // Operators
    Tensor log() const;
    Tensor exp() const;

    // Tensor operator-() const;
    // Tensor operator==(Tensor other) const;
    //
    // Tensor operator+(double scalar) const;
    // Tensor operator-(double scalar) const;
    // Tensor operator*(double scalar) const;
    // Tensor operator/(double scalar) const;
    //
    // Tensor operator+(Tensor other) const;
    // Tensor operator*(Tensor other) const;
    // Tensor operator-(Tensor other) const;
    // Tensor operator/(Tensor other) const;

    float get(const std::vector<size_t> &indices);
    friend std::ostream &operator<<(std::ostream &os, const CPUTensor &t);

    // Device
    Tensor to(Device dev) const;

  private:
    Tensor(const Shape &shape, Device device, EmptyTag);
    Tensor(TensorVariant &tensor, Device device);
    Tensor(const Shape &shape, float value, Device device);
    Tensor(const Shape &shape, std::mt19937 rng, Device device);

    static TensorVariant tensor_from_data(const std::vector<float> &data, Device device);
    static TensorVariant tensor_from_data(const std::vector<std::vector<float>> &data,
                                          Device device);
    static TensorVariant tensor_from_shape(const Shape &shape, Device device);
    static TensorVariant filled_tensor(const Shape &shape, float value, Device device);
    static TensorVariant random_tensor(const Shape &shape, std::mt19937 rng, Device device);
};

std::ostream &operator<<(std::ostream &os, const Tensor &t);

#endif
