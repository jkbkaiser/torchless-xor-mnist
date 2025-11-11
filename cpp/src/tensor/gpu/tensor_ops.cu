#include <cmath>

#include <torchless/tensor/tensor_gpu.h>

template <typename Func>
__global__ void map_kernel(float *res, float *data, size_t size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = func(data[idx]);
    }
}

template <typename Func>
GPUTensor GPUTensor::map(Func func) const {
    GPUTensor result(shape_);
    int blocks = (size_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    map_kernel<<<blocks, THREADS_PER_BLOCK>>>(result.data_, data_, size_, func);
    cudaDeviceSynchronize();
    return result;
}

GPUTensor GPUTensor::log() const {
    return this->map([] __device__ (float x) { return logf(x); });
}

GPUTensor GPUTensor::exp() const {
    return this->map([] __device__ (float x) { return expf(x); });
}
