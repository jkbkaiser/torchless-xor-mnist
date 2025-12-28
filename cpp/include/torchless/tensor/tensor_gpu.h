#ifndef TORCHLESS_TENSOR_GPU_H
#define TORCHLESS_TENSOR_GPU_H

#include <functional>
#include <ostream>
#include <random>
#include <vector>

#include <torchless/shape.h>
#include <torchless/tensor/tensor_base.h>

class CPUTensor;

inline constexpr int THREADS_PER_BLOCK = 256;

class GPUTensor : public BaseTensor {
  public:
    float *data_;
    size_t size_;

    // Factories
    GPUTensor(const Shape &shape);
    GPUTensor(const std::vector<float> &values);
    GPUTensor(const std::vector<std::vector<float>> &values);
    static GPUTensor filled(const Shape &shape, float value);
    static GPUTensor rand(const Shape &shape, std::mt19937 &rng);

    // Ops
    GPUTensor log() const;
    GPUTensor exp() const;
    GPUTensor operator-() const;

    GPUTensor operator+(float scalar) const;
    GPUTensor operator-(float scalar) const;
    GPUTensor operator*(float scalar) const;
    GPUTensor operator/(float scalar) const;

    GPUTensor operator+(const GPUTensor &other) const;
    GPUTensor operator==(const GPUTensor &other) const;

    float get(const std::vector<size_t> &indices) const;
    void set(const std::vector<size_t> &indices, float val);
    CPUTensor toCPU() const;

    template <typename Func> GPUTensor map(const Func func) const;

  private:
    template <typename Func> GPUTensor map2(const GPUTensor &other, const Func func) const;
};

GPUTensor operator/(float scalar, const GPUTensor &tensor);

std::ostream &operator<<(std::ostream &os, const GPUTensor &t);

#endif
