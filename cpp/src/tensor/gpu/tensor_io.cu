#include <torchless/tensor/tensor_cpu.h>
#include <torchless/tensor/tensor_gpu.h>

CPUTensor GPUTensor::toCPU() const {
    CPUTensor t(shape_);
    cudaMemcpy(t.data_.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return t;
}

std::ostream &operator<<(std::ostream &os, const GPUTensor &t) {
    auto cpuT = t.toCPU();
    os << cpuT;
    return os;
}
