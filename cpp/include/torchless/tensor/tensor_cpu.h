#ifndef TORCHLESS_TENSOR_CPU_H
#define TORCHLESS_TENSOR_CPU_H

#include <functional>
#include <ostream>
#include <random>
#include <vector>

#include <torchless/shape.h>
#include <torchless/tensor/tensor_base.h>

class GPUTensor;

class CPUTensor : public BaseTensor {
  public:
    std::vector<float> data_;

    // Factories
    CPUTensor(const Shape &shape);
    CPUTensor(const std::vector<float> &values);
    CPUTensor(const std::vector<std::vector<float>> &values);
    static CPUTensor filled(const Shape &shape, float value);
    static CPUTensor rand(const Shape &shape, std::mt19937 &rng);

    // Ops
    CPUTensor log() const;
    CPUTensor exp() const;

    float get(const std::vector<size_t> &indices);
    GPUTensor toGPU() const;

  private:
    CPUTensor map(const std::function<float(float)> &func) const;
};

std::ostream &operator<<(std::ostream &os, const CPUTensor &t);

// Scalar operators
// template <Device D> Tensor<D> operator+(double scalar, const Tensor<D> &t);
// template <Device D> Tensor<D> operator-(double scalar, const Tensor<D> &t);
// template <Device D> Tensor<D> operator*(double scalar, const Tensor<D> &t);
//
// template <Device D> Tensor<D> operator/(double scalar, const Tensor<D> &t);
//
// template <Device D> Tensor<D> stack(const std::vector<Tensor<D>> &tensors, int axis = 0);

#endif

// #ifndef TORCHLESS_TENSOR_CPU_H
// #define TORCHLESS_TENSOR_CPU_H
//
// // #include <functional>
// // #include <iostream>
// // #include <random>
// #include <vector>
//
// #include <torchless/tensor/tensor_base.h>
//
// class CPUTensor : public BaseTensor {
//   public:
//     std::vector<double> data_;
//
//     // static Tensor<D> empty(const std::vector<size_t> &shape_);
//
//     // friend std::ostream &operator<<(std::ostream &os, const Tensor<D> &t);
//     //
//     // Tensor<D> map(const std::function<double(double)> &func) const;
//     //
//     // double &at(const std::vector<size_t> &indices);
//     // double at(const std::vector<size_t> &indices) const;
//     // double item() const;
//     //
//     CPUTensor(const std::vector<size_t> &shape);
//     // Tensor(const std::vector<double> &vec);
//     // Tensor(const std::vector<double> &vec, const std::vector<size_t> &shape_);
//     // Tensor(const std::vector<std::vector<double>> &vec);
//
//     // static CPUTensor empty(const std::vector<size_t> &shape);
//     // static Tensor<D> zeros(const std::vector<size_t> &shape);
//     // static Tensor<D> ones(const std::vector<size_t> &shape);
//     // static Tensor<D> filled(const std::vector<size_t> &shape, double value);
//     // static Tensor<D> rand(const std::vector<size_t> &shape, std::mt19937 rng);
//     //
//     // Tensor<D> squeeze() const;
//     // Tensor<D> add_dim(int dim) const;
//     //
//     // Tensor<D> log() const;
//     // Tensor<D> exp() const;
//     // Tensor<D> max() const;
//     // Tensor<D> sum() const;
//     // Tensor<D> sum(int axis) const;
//     //
//     // Tensor<D> operator-() const;
//     // Tensor<D> operator==(Tensor<D> other) const;
//     //
//     // Tensor<D> operator+(double scalar) const;
//     // Tensor<D> operator-(double scalar) const;
//     // Tensor<D> operator*(double scalar) const;
//     // Tensor<D> operator/(double scalar) const;
//     //
//     // Tensor<D> operator+(Tensor<D> other) const;
//     // Tensor<D> operator*(Tensor<D> other) const;
//     // Tensor<D> operator-(Tensor<D> other) const;
//     // Tensor<D> operator/(Tensor<D> other) const;
//     //
//     // Tensor<D> transpose() const;
//     // Tensor<D> dot(Tensor<D> other) const;
//
//     // Matrix multiplication
//     // We support the following dimensions:
//     //   [N] * [N]             (dot product)
//     //   [M, N] * [N, K]       (matrix-matrix multiplication)
//     //   [N] * [N, K]          (matrix-vector multiplication)
//     //   [N, K] * [K]          (matrix-vector multiplication)
//     //   [O, M, N] * [L, N, K] (batched matrix-matrix multiplication)
//     //   [O, M, N] * [N, K]    (broadcast batched matrix-matrix multiplication)
//     //   [M, N] * [L, N, K]    (broadcast batched matrix-matrix multiplication)
//     // Tensor<D> matmul(Tensor<D> other) const;
//
//     // TODO
//     // Implement all matrix multiplications
//     // Compound operators (+=)
//     // Gradients
// };
//
// // Scalar operators
// // template <Device D> Tensor<D> operator+(double scalar, const Tensor<D> &t);
// // template <Device D> Tensor<D> operator-(double scalar, const Tensor<D> &t);
// // template <Device D> Tensor<D> operator*(double scalar, const Tensor<D> &t);
// //
// // template <Device D> Tensor<D> operator/(double scalar, const Tensor<D> &t);
// //
// // template <Device D> Tensor<D> stack(const std::vector<Tensor<D>> &tensors, int axis = 0);
//
// #endif
