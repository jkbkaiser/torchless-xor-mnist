#include <torchless/tensor/tensor_gpu.h>

float GPUTensor::get(const std::vector<size_t> &indices) {
    size_t idx = 0;
    size_t stride = 1;

    // flatten indices to 1D index
    for (int i = indices.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape_.dims_[i];
    }

    float value;
    cudaMemcpy(&value, data_ + idx, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}
