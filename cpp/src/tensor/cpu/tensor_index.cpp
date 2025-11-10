#include <torchless/tensor/tensor_cpu.h>

float CPUTensor::get(const std::vector<size_t> &indices) {
    int offset = 0;
    int stride = 1;

    for (int i = (int)shape_.size() - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape_[i];
    }

    return data_[offset];
}
