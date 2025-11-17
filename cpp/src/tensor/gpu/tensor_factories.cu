#include <iostream>
#include <random>
#include <vector>

#include <torchless/tensor/tensor_cpu.h>
#include <torchless/tensor/tensor_gpu.h>

// Constructors
GPUTensor::GPUTensor(const Shape &shape)
    : BaseTensor(shape) {
    size_ = shape.numel();
    cudaMalloc(&data_, size_ * sizeof(float));
}

GPUTensor::GPUTensor(const std::vector<float> &values)
    : BaseTensor({values.size()}), size_(values.size()) {
    cudaMalloc(&data_, size_ * sizeof(float));
    cudaError_t err =  cudaMemcpy(
        data_,
        values.data(),
        size_ * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

GPUTensor::GPUTensor(const std::vector<std::vector<float>> &values)
    : BaseTensor({values.size(), values[0].size()}),
      size_(values.size() * values[0].size()) {
    cudaMalloc(&data_, size_ * sizeof(float));

    // Flatten the 2D vector first
    std::vector<float> flat;
    flat.reserve(size_);
    for (auto &row : values)
        flat.insert(flat.end(), row.begin(), row.end());

    cudaError_t err = cudaMemcpy(
        data_,
        flat.data(),
        size_ * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

__global__ void fill_kernel(float* data, size_t size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

GPUTensor GPUTensor::filled(const Shape &shape, float value) {
    GPUTensor t(shape);
    int blocks = (t.size_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<blocks, THREADS_PER_BLOCK>>>(t.data_, t.size_, value);
    cudaDeviceSynchronize();
    return t;
}

GPUTensor GPUTensor::rand(const Shape &shape, std::mt19937 &rng) {
    return CPUTensor::rand(shape, rng).toGPU();
}



// Destructor
// GPUTensor::~GPUTensor() = default;

// Move constructors/operators
// GPUTensor::GPUTensor(GPUTensor &&other) noexcept
//     : BaseTensor(std::move(other)), data_(other.data_), size_(other.size_) {
//     other.data_ = nullptr;
//     other.size_ = 0;
// }

// GPUTensor &GPUTensor::operator=(GPUTensor &&other) noexcept {
//     if (this != &other) {
//         data_ = other.data_;
//         size_ = other.size_;
//         other.data_ = nullptr;
//         other.size_ = 0;
//     }
//     return *this;
// }

// // Stream operator
// std::ostream &operator<<(std::ostream &os, const GPUTensor &/*t*/) {
//     os << "<GPUTensor>";
//     return os;
// }

