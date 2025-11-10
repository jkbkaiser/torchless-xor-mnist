#include <iomanip>
#include <ostream>

#include <torchless/shape.h>
#include <torchless/tensor/tensor_cpu.h>
#include <torchless/tensor/tensor_gpu.h>

void print_recursive(std::ostream &os, const std::vector<float> &data, const Shape &shape, int dim,
                     int offset, int &rows, int &dotted_rows, const std::string &indent = "") {
    if (rows == 0) {
        if (dotted_rows > 0) {
            os << indent << "...\n";
            dotted_rows--;
        }
        return;
    } else if (dim == (int)shape.size() - 1) {
        os << indent << "[ ";
        for (size_t i = 0; i < shape[dim]; ++i) {
            os << data[offset + i] << " ";
        }
        os << "]\n";
        rows--;
    } else {
        os << indent << "[\n";
        int step = 1;
        for (size_t d = dim + 1; d < shape.size(); ++d) {
            step *= shape[d];
        }
        for (size_t i = 0; i < shape[dim]; ++i) {
            print_recursive(os, data, shape, dim + 1, offset + i * step, rows, dotted_rows,
                            indent + "  ");
        }
        os << indent << "]\n";
    }
}

std::ostream &operator<<(std::ostream &os, const CPUTensor &t) {
    int rows = 10;
    int dotted_rows = 3;
    os << std::fixed << std::setprecision(3);
    print_recursive(os, t.data_, t.shape_, 0, 0, rows, dotted_rows);
    return os;
}

GPUTensor CPUTensor::toGPU() {
    GPUTensor t(shape_);

    cudaMemcpy(t.data_, data_.data(), data_.size() * sizeof(float), cudaMemcpyHostToDevice);

    return t;
}
