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

template <typename Func>
__global__ void map2_kernel(float *res, float *data_a, float *data_b, size_t size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = func(data_a[idx], data_b[idx]);
    }
}

template <typename Func>
GPUTensor GPUTensor::map2(const GPUTensor &other, const Func func) const {
    GPUTensor result(shape_);
    int blocks = (size_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    map2_kernel<<<blocks, THREADS_PER_BLOCK>>>(result.data_, data_, other.data_, size_, func);
    cudaDeviceSynchronize();
    return result;
}

GPUTensor GPUTensor::log() const {
    return this->map([] __device__ (float x) { return logf(x); });
}

GPUTensor GPUTensor::exp() const {
    return this->map([] __device__ (float x) { return expf(x); });
}

GPUTensor GPUTensor::operator-() const {
    return this->map([] __device__ (float x) { return -x; });
}

GPUTensor GPUTensor::operator+(float scalar) const {
    return this->map([scalar] __device__ (float x) { return x + scalar; });
}

GPUTensor GPUTensor::operator-(float scalar) const {
    return this->map([scalar] __device__ (float x) { return x - scalar; });
}

GPUTensor GPUTensor::operator*(float scalar) const {
    return this->map([scalar] __device__ (float x) { return x * scalar; });
}

GPUTensor GPUTensor::operator/(float scalar) const {
    return this->map([scalar] __device__ (float x) { return x / scalar; });
}

GPUTensor GPUTensor::operator==(const GPUTensor &other) const {
    return this->map2(other, [] __device__ (float a, float b) { return a == b; });
}

GPUTensor operator/(float scalar, const GPUTensor &tensor) {
    return tensor.map([scalar] __device__ (float x) { return scalar / x; });
}

