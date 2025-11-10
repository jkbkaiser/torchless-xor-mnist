#include <iostream>
#include <random>
#include <vector>

#include <torchless/tensor/tensor_cpu.h>
#include <torchless/tensor/tensor_gpu.h>

CPUTensor toCPU() {
    CPUTensor t(shape_);
    cudaMemcpy(t.data, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return t;
}


std::ostream &operator<<(std::ostream &os, const GPUTensor &t) {
    auto cpuT = t.toCPU();
    os << cpuT;
    return os;
}
