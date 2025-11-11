// Element-wise operations and broadcasting
//
// Scalar operations:
//   - operator+(Tensor, scalar), operator+(scalar, Tensor)
//   - operator-(Tensor, scalar), operator-(scalar, Tensor), operator-(Tensor) [unary]
//   - operator*(Tensor, scalar), operator*(scalar, Tensor)
//   - operator/(Tensor, scalar), operator/(scalar, Tensor)
//
// Tensor-tensor operations with broadcasting:
//   - operator+(Tensor, Tensor)
//   - operator-(Tensor, Tensor)
//   - operator*(Tensor, Tensor)  [element-wise]
//   - operator/(Tensor, Tensor)
//   - operator==(Tensor, Tensor)
//
// Compound assignment operators:
//   - operator+=(Tensor), operator+=(scalar)
//   - operator-=(Tensor), operator-=(scalar)
//   - operator*=(Tensor), operator*=(scalar)
//   - operator/=(Tensor), operator/=(scalar)
//
// Broadcasting helpers:
//   - broadcast_shape()
//   - broadcast_idx()
//   - all_indices()

#include <functional>

#include <torchless/tensor.h>

CPUTensor CPUTensor::map(const std::function<float(float)> &func) const {
    CPUTensor result(shape_);
    std::transform(data_.begin(), data_.end(), result.data_.begin(), func);
    return result;
}

CPUTensor CPUTensor::log() const {
    return this->map([](float x) { return std::log(x); });
}

CPUTensor CPUTensor::exp() const {
    return this->map([](float x) { return std::exp(x); });
}

// Tensor Tensor::operator+(double scalar) const {
//     return this->map([scalar](double x) { return x + scalar; });
// }
//
// Tensor operator+(double scalar, const Tensor &t) { return t + scalar; }
//
// Tensor Tensor::operator-(double scalar) const {
//     return this->map([scalar](double x) { return x - scalar; });
// }
//
// Tensor Tensor::operator-() const {
//     return this->map([](double x) { return -x; });
// }
//
// Tensor operator-(double scalar, const Tensor &t) {
//     return t.map([scalar](double x) { return scalar - x; });
// }
//
// Tensor Tensor::operator*(double scalar) const {
//     return this->map([scalar](double x) { return x * scalar; });
// }
//
// Tensor operator*(double scalar, const Tensor &t) { return t * scalar; }
//
// Tensor Tensor::operator/(double scalar) const {
//     if (scalar == 0.0) {
//         throw std::runtime_error("Division by zero.");
//     }
//     return this->map([scalar](double x) { return x / scalar; });
// }
//
// Tensor operator/(double scalar, const Tensor &t) {
//     return t.map([scalar](double x) {
//         if (x == 0.0) {
//             throw std::runtime_error("Division by zero.");
//         }
//         return scalar / x;
//     });
// }
//
// std::vector<size_t> broadcast_shape(const std::vector<size_t> &shape1,
//                                     const std::vector<size_t> &shape2) {
//     size_t n1 = shape1.size(), n2 = shape2.size();
//     size_t n = std::max(n1, n2);
//
//     std::vector<size_t> result(n);
//
//     for (size_t i = 0; i < n; ++i) {
//         int dim1 = (i < n - n1) ? 1 : shape1[i - (n - n1)];
//         int dim2 = (i < n - n2) ? 1 : shape2[i - (n - n2)];
//
//         if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
//             result[i] = std::max(dim1, dim2);
//         else
//             throw std::runtime_error("Shapes not broadcastable");
//     }
//
//     return result;
// }
//
// std::vector<std::vector<size_t>> all_indices(const std::vector<size_t> &shape) {
//     std::vector<std::vector<size_t>> indices;
//
//     if (shape.empty()) {
//         return indices;
//     }
//
//     std::vector<size_t> current(shape.size(), 0);
//
//     while (true) {
//         indices.push_back(current);
//
//         int dim = shape.size() - 1;
//         while (dim >= 0) {
//             current[dim]++;
//             if (current[dim] < shape[dim]) {
//                 break;
//             }
//             current[dim] = 0;
//             dim--;
//         }
//
//         if (dim < 0) {
//             break;
//         }
//     }
//
//     return indices;
// }
//
// std::vector<size_t> broadcast_idx(std::vector<size_t> idx, const std::vector<size_t> &shape) {
//     std::vector<size_t> broadcasted_idx(shape.size());
//
//     for (size_t i = 1; i <= shape.size(); ++i) {
//         broadcasted_idx[shape.size() - i] = idx[idx.size() - i] % shape[shape.size() - i];
//     }
//
//     return broadcasted_idx;
// }
//
// Tensor Tensor::operator+(Tensor other) const {
//     auto out_shape = broadcast_shape(this->shape, other.shape);
//     Tensor out(out_shape);
//
//     for (auto idx : all_indices(out_shape)) {
//         auto idx1 = broadcast_idx(idx, this->shape);
//         auto idx2 = broadcast_idx(idx, other.shape);
//
//         out.at(idx) = this->at(idx1) + other.at(idx2);
//     }
//
//     return out;
// }
//
// Tensor Tensor::operator*(Tensor other) const {
//     auto out_shape = broadcast_shape(this->shape, other.shape);
//     Tensor out(out_shape);
//
//     for (auto idx : all_indices(out_shape)) {
//         auto idx1 = broadcast_idx(idx, this->shape);
//         auto idx2 = broadcast_idx(idx, other.shape);
//
//         out.at(idx) = this->at(idx1) * other.at(idx2);
//     }
//
//     return out;
// }
//
// Tensor Tensor::operator-(Tensor other) const { return *this + (-1.0 * other); }
//
// Tensor Tensor::operator/(Tensor other) const {
//     auto out_shape = broadcast_shape(this->shape, other.shape);
//     Tensor out(out_shape);
//
//     for (auto idx : all_indices(out_shape)) {
//         auto idx1 = broadcast_idx(idx, this->shape);
//         auto idx2 = broadcast_idx(idx, other.shape);
//
//         out.at(idx) = this->at(idx1) / other.at(idx2);
//     }
//
//     return out;
// }
