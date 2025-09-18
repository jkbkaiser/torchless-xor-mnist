#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "tensor.h"


Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_) {
    int size = std::accumulate(shape.begin(), shape.end(), 1,
                    std::multiplies<int>());
    data.resize(size);
}


void print_recursive(
    std::ostream& os, 
    const std::vector<double>& data, 
    const std::vector<int>& shape,
    int dim,
    int offset,
    int& rows,
    int& dotted_rows,
    const std::string& indent = ""
) {
    if (rows == 0) {
        if (dotted_rows > 0) {
            os << indent << "...\n";
            dotted_rows--;
        }
        return;
    } else if (dim == (int)shape.size() - 1) {
        os << indent << "[ ";
        for (int i = 0; i < shape[dim]; ++i) {
            os << data[offset + i] << " ";
        }
        os << "]\n";
        rows--;
    } else {
        os << indent << "[\n";
        int step = 1;
        for (int d = dim + 1; d < (int)shape.size(); ++d) {
            step *= shape[d];
        }
        for (int i = 0; i < shape[dim]; ++i) {
            print_recursive(os, data, shape, dim + 1, offset + i * step, rows, dotted_rows, indent + "  ");
        }
        os << indent << "]\n";
    }
}


std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    int rows = 10;
    int dotted_rows = 3;
    os << std::fixed << std::setprecision(3);
    print_recursive(os, t.data, t.shape, 0, 0, rows, dotted_rows);
    return os;
}


Tensor Tensor::from_vec(const std::vector<std::vector<double>>& vec) {
    Tensor t({(int) vec.size(), (int) vec[0].size()});

    for (int i = 0; i < vec.size(); ++i) {
        for (int j = 0; j < vec[i].size(); ++j) {
            t.at({i, j}) = vec[i][j];
        }
    }

    return t;
}


Tensor Tensor::from_vec(const std::vector<double>& vec) {
    Tensor t({(int) vec.size()});

    for (int i = 0; i < vec.size(); ++i) {
        t.at({i}) = vec[i];
    }

    return t;
}


Tensor Tensor::filled(const std::vector<int>& shape, double value) {
    Tensor t(shape);
    std::fill(t.data.begin(), t.data.end(), value);
    return t;
}


Tensor Tensor::zeros(const std::vector<int>& shape) {
    return Tensor::filled(shape, 0.0);
}


Tensor Tensor::ones(const std::vector<int>& shape) {
    return Tensor::filled(shape, 1.0);
}


Tensor Tensor::rand(const std::vector<int>& shape) {
    Tensor t(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(0.0, 1.0);
    std::generate(t.data.begin(), t.data.end(), std::bind(d, gen));
    return t;
}


Tensor Tensor::eye(const std::vector<int>& shape) {
    Tensor t = Tensor::zeros(shape);
    for (int i = 0; i < shape[0]; ++i) {
        t.data[i*shape[1] + i] = 1.0;
    }
    return t;
}


Tensor Tensor::transpose() const {
    if (this->shape.size() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D matrices");
    }

    Tensor t = Tensor({this->shape[1], this->shape[0]});

    for (int i = 0; i < this->shape[0]; ++i) {
        for (int j = 0; j < this->shape[1]; ++j) {
            t.at({j, i}) = this->at({i, j});
        }
    }

    return t;
}

Tensor Tensor::add_dim(const int dim) const {
    Tensor result(this->shape);
    result.data = this->data;
    result.shape.insert(result.shape.begin() + dim, 1);
    return result;
}

Tensor Tensor::map(const std::function<double(double)>& func) const {
    Tensor result(this->shape);
    std::transform(this->data.begin(), this->data.end(), result.data.begin(), func);
    return result;
}


double& Tensor::at(const std::vector<int>& indices) {
    int offset = 0;
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape[i];
    }
    return data[offset];
}


Tensor Tensor::max() const {
    double max = data[0];
    for (int i = 1; i < data.size(); ++i) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return Tensor::filled({1}, max);
}


// Tensor Tensor::max(int dim) const {
//     // Handle negative dim: convert -1 to last dim index
//     if (dim < 0) dim = shape.size() + dim;
//     if (dim != shape.size() - 1) {
//         throw std::runtime_error("Only max over last dimension (-1) is implemented");
//     }
//
//     int rows = 1;
//     for (int i = 0; i < dim; ++i) {
//         rows *= shape[i];
//     }
//     int last_dim = shape[dim];
//
//     Tensor result({rows});
//
//     for (int i = 0; i < rows; ++i) {
//         double max_val = data[i * last_dim];  // first element in the last dim slice
//         for (int j = 1; j < last_dim; ++j) {
//             double val = data[i * last_dim + j];
//             if (val > max_val) max_val = val;
//         }
//         result.at({i}) = max_val;
//     }
//
//     return result;
// }


Tensor Tensor::log() const {
    Tensor result(this->shape);
    std::transform(this->data.begin(), this->data.end(), result.data.begin(), [](double x) { return std::log(x); });
    return result;
}

Tensor Tensor::exp() const {
    Tensor result(this->shape);
    std::transform(this->data.begin(), this->data.end(), result.data.begin(), [](double x) { return std::exp(x); });
    return result;
}


Tensor Tensor::sum() const {
    double sum = 0.0;
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return Tensor::filled({1}, sum);
}


// Tensor Tensor::sum(int dim) const {
//     // Handle negative dim
//     if (dim < 0) dim = shape.size() + dim;
//     
//     if (dim != shape.size() - 1) {
//         throw std::runtime_error("Only sum over last dimension (-1) is implemented");
//     }
//
//     int outer_size = 1;
//     for (int i = 0; i < dim; ++i) {
//         outer_size *= shape[i];
//     }
//     int last_dim = shape[dim];
//
//     // Result shape = input shape without last dimension
//     std::vector<int> result_shape(shape.begin(), shape.end());
//     result_shape.erase(result_shape.begin() + dim);
//     Tensor result(result_shape);
//
//     for (int i = 0; i < outer_size; ++i) {
//         double sum_val = 0.0;
//         for (int j = 0; j < last_dim; ++j) {
//             sum_val += data[i * last_dim + j];
//         }
//         result.at({i}) = sum_val;
//     }
//
//     return result;
// }


double Tensor::item() const {
    return data[0];
}


double Tensor::at(const std::vector<int>& indices) const {
    int offset = 0;
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape[i];
    }
    return data[offset];
}


bool Tensor::operator==(Tensor other) const {
    if (this->shape != other.shape) return false;

    for (int i = 0; i < this->data.size(); ++i) {
        if (this->data[i] != other.data[i]) return false;
    }

    return true;
}


bool Tensor::operator!=(Tensor other) const {
    return !(*this == other);
}


Tensor Tensor::operator+(double scalar) const {
    return this->map([scalar](double x) { return x + scalar; });
}


Tensor operator+(double scalar, const Tensor& t) {
    return t + scalar;
}


Tensor Tensor::operator-(double scalar) const {
    return this->map([scalar](double x) { return x - scalar; });
}


Tensor Tensor::operator-() const {
    return this->map([](double x) { return -x; });
}


Tensor operator-(double scalar, const Tensor& t) {
    return t - scalar;
}


Tensor Tensor::operator*(double scalar) const {
    return this->map([scalar](double x) { return x * scalar; });
}


Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}


Tensor Tensor::operator/(double scalar) const {
    return this->map(
        [scalar](double x) {
            if (x == 0.0) {
                throw std::runtime_error("Division by zero.");
            }
            return x / scalar;
        }
    );
}


Tensor operator/(double scalar, const Tensor& t) {
    return t.map(
        [scalar](double x) {
            if (x == 0.0) {
                throw std::runtime_error("Division by zero.");
            }
            return scalar / x;
        });
}


std::vector<int> broadcast_shape(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    int n1 = shape1.size(), n2 = shape2.size();
    int n = std::max(n1, n2);

    std::vector<int> result(n);

    for (int i = 0; i < n; ++i) {
        int dim1 = (i < n - n1) ? 1 : shape1[i - (n - n1)];
        int dim2 = (i < n - n2) ? 1 : shape2[i - (n - n2)];

        if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
            result[i] = std::max(dim1, dim2);
        else
            throw std::runtime_error("Shapes not broadcastable");
    }

    return result;
}


std::vector<std::vector<int>> all_indices(const std::vector<int>& shape) {
    std::vector<std::vector<int>> indices;

    if (shape.empty()) {
        return indices;
    }

    std::vector<int> current(shape.size(), 0);

    while (true) {
        indices.push_back(current);

        int dim = shape.size() - 1;
        while (dim >= 0) {
            current[dim]++;
            if (current[dim] < shape[dim]) {
                break;
            }
            current[dim] = 0;
            dim--;
        }

        if (dim < 0) {
            break;
        }
    }

    return indices;
}

std::vector<int> broadcast_idx(std::vector<int> idx, const std::vector<int>& shape) {
    std::vector<int> broadcasted_idx(shape.size());

    for (int i = 1; i <= shape.size(); ++i) {
        broadcasted_idx[shape.size() - i] = idx[idx.size() - i] % shape[shape.size() - i];
    }

    return broadcasted_idx;
}

Tensor Tensor::operator+(Tensor other) const {
    auto out_shape = broadcast_shape(this->shape, other.shape);
    Tensor out(out_shape);

    for (auto idx : all_indices(out_shape)) {
        auto idx1 = broadcast_idx(idx, this->shape);
        auto idx2 = broadcast_idx(idx, other.shape);

        out.at(idx) = this->at(idx1) + other.at(idx2);
    }

    return out;
}


Tensor Tensor::operator-(Tensor other) const {
    return *this + (-1.0 * other);
}


Tensor Tensor::operator/(Tensor other) const {
    auto out_shape = broadcast_shape(this->shape, other.shape);
    Tensor out(out_shape);

    for (auto idx : all_indices(out_shape)) {
        auto idx1 = broadcast_idx(idx, this->shape);
        auto idx2 = broadcast_idx(idx, other.shape);

        out.at(idx) = this->at(idx1) / other.at(idx2);
    }

    return out;
}


Tensor Tensor::dot(Tensor other) const {
    if (this->shape.size() != 1 || other.shape.size() != 1) {
        throw std::runtime_error("Dot product is only defined for 1D tensors.");
    }
    if (this->shape[0] != other.shape[0]) {
        throw std::runtime_error("Dot product is only defined for tensors with compatible dimensions.");
    }

    double result = 0.0;
    for (int i = 0; i < this->shape[0]; ++i) {
        result += this->at({i}) * other.at({i});
    }

    return Tensor::filled({1}, result);
}


Tensor Tensor::matmul(Tensor other) const {
    // Dot product
    if (this->shape.size() == 1 && other.shape.size() == 1) {
        return this->dot(other);
    }

    // Matrix-matrix multiplication
    if (this->shape.size() == 2 && other.shape.size() == 2) {
        if (this->shape[1] != other.shape[0]) {
            throw std::runtime_error("Matrix-matrix multiplication is only defined for compatible dimensions.");
        }

        auto result = Tensor({this->shape[0], other.shape[1]});

        for (int i = 0; i < this->shape[0]; ++i) {
            for (int j = 0; j < other.shape[1]; ++j) {
                double sum = 0.0;
                for (int k = 0; k < this->shape[1]; ++k) {
                    sum += this->at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }

        return result;
    }

    // Matrix-vector multiplication
    if (this->shape.size() == 2 && other.shape.size() == 1) {
        if (this->shape[1] != other.shape[0]) {
            throw std::runtime_error("Matrix-vector multiplication is only defined for compatible dimensions.");
        }

        auto result = Tensor({this->shape[0]});

        for (int i = 0; i < this->shape[0]; ++i) {
            double sum = 0.0;
            for (int k = 0; k < this->shape[1]; ++k) {
                sum += this->at({i, k}) * other.at({k});
            }
            result.at({i}) = sum;
        }

        return result;
    }

    // Matrix-vector multiplication
    if (this->shape.size() == 1 && other.shape.size() == 2) {
        if (this->shape[0] != other.shape[0]) {
            throw std::runtime_error("Matrix-vector multiplication is only defined for compatible dimensions.");
        }

        auto result = Tensor({other.shape[1]});

        for (int j = 0; j < other.shape[1]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < this->shape[0]; ++k) {
                sum += this->at({k}) * other.at({k, j});
            }
            result.at({j}) = sum;
        }

        return result;
    }

    throw std::runtime_error("Matrix multiplication is not yet implemented for this input.");
}


Tensor stack(const std::vector<Tensor>& tensors) {
    if (tensors.size() == 0) {
        throw std::runtime_error("Cannot stack an empty list of tensors.");
    }

    if (tensors.size() == 1) {
        return tensors[0];
    }

    if (tensors[0].shape.size() != 1) {
        throw std::runtime_error("Stacking is only supported for tensors with one dimension.");
    }

    for (int i = 0; i < tensors.size(); ++i) {
        if (tensors[i].shape != tensors[0].shape) {
            throw std::runtime_error("Tensors must have the same shape to be stacked.");
        }
    }

    std::vector<int> shape = {(int) tensors.size(), tensors[0].shape[0]};

    Tensor result = Tensor(shape);

    for (int i = 0; i < tensors[0].shape[0]; ++i) {
        for (int j = 0; j < tensors.size(); ++j) {
            result.at({j, i}) = tensors[j].at({i});
        }
    }

    return result;
}
