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

    float get(const std::vector<size_t> &indices);
    CPUTensor toCPU() const;

  private:
    template <typename Func> GPUTensor map(Func func) const;
};

std::ostream &operator<<(std::ostream &os, const GPUTensor &t);

#endif
