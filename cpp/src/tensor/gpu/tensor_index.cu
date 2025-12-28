#include <torchless/tensor/tensor_gpu.h>

float GPUTensor::get(const std::vector<size_t> &indices) const {
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

void GPUTensor::set(const std::vector<size_t> &indices, float value) {
    if (indices.size() != shape_.dims_.size()) {
        throw std::invalid_argument("Index rank mismatch");
    }

    // flatten indices to 1D
    size_t idx = 0;
    size_t stride = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape_.dims_[i];
    }

    // copy the value from host to device
    cudaError_t err = cudaMemcpy(data_ + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
    }
}
