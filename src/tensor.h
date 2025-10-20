#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <iostream>
#include <vector>

class Tensor {
  public:
    std::vector<size_t> shape;
    std::vector<double> data;

    Tensor(const std::vector<size_t> &shape_);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);

    Tensor map(const std::function<double(double)> &func) const;

    double &at(const std::vector<size_t> &indices);
    double at(const std::vector<size_t> &indices) const;
    double item() const;

    static Tensor from_vec(const std::vector<double> &vec);
    static Tensor from_vec(const std::vector<double> &vec, const std::vector<size_t> &shape_);
    static Tensor from_vec(const std::vector<std::vector<double>> &vec);

    static Tensor zeros(const std::vector<size_t> &shape);
    static Tensor ones(const std::vector<size_t> &shape);
    static Tensor filled(const std::vector<size_t> &shape, double value);
    static Tensor rand(const std::vector<size_t> &shape);

    Tensor transpose() const;
    Tensor log() const;
    Tensor exp() const;
    Tensor max() const;
    Tensor sum() const;
    Tensor sum(int axis) const;
    Tensor squeeze() const;
    Tensor operator-() const;
    Tensor add_dim(int dim) const;

    // Scalar operators
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;

    Tensor operator==(Tensor other) const;
    Tensor operator+(Tensor other) const;
    Tensor operator*(Tensor other) const;
    Tensor operator-(Tensor other) const;
    Tensor operator/(Tensor other) const;
    // Tensor operator*(Tensor other) const;

    // Dot product
    Tensor dot(Tensor other) const;

    // Matrix multiplication
    // We support the following dimensions:
    //   [N] * [N]             (dot product)
    //   [M, N] * [N, K]       (matrix-matrix multiplication)
    //   [N] * [N, K]          (matrix-vector multiplication)
    //   [N, K] * [K]          (matrix-vector multiplication)
    //   [O, M, N] * [L, N, K] (batched matrix-matrix multiplication)
    //   [O, M, N] * [N, K]    (broadcast batched matrix-matrix multiplication)
    //   [M, N] * [L, N, K]    (broadcast batched matrix-matrix multiplication)
    Tensor matmul(Tensor other) const;

    // TODO
    // Implement all matrix multiplications
    // Compound operators (+=)
    // Gradients
};

// Scalar operators
Tensor operator+(double scalar, const Tensor &t);
Tensor operator-(double scalar, const Tensor &t);
Tensor operator*(double scalar, const Tensor &t);
Tensor operator/(double scalar, const Tensor &t);

Tensor stack(const std::vector<Tensor> &tensors, int axis = 0);

#endif
