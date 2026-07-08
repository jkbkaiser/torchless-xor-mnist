#include <torchless/tensor.h>

#include <functional>
#include <iostream>

void test_factories_1D(Device dev);
void test_factories_2D(Device dev);
void test_unary_ops(Device dev);
void test_tensor_scalar_ops(Device dev);
void test_tensor_tensor_ops(Device dev);

void run_test(const std::string &name, const std::function<void(Device)> &test_func) {
    std::cout << "Running " << name << " on CPU ..." << std::flush;
    test_func(Device::CPU);
    std::cout << "\t✅ " << std::endl;

    std::cout << "Running " << name << " on GPU ..." << std::flush;
    test_func(Device::GPU);
    std::cout << "\t✅ " << std::endl;
}

// Macro to automatically pass the function name as a string
#define RUN_TEST(func) run_test(#func, func)

int main() {
    std::cout << "\nRunning all tests\n---------------" << std::endl;
    RUN_TEST(test_factories_1D);
    RUN_TEST(test_factories_2D);
    RUN_TEST(test_unary_ops);
    RUN_TEST(test_tensor_tensor_ops);
    RUN_TEST(test_tensor_scalar_ops);

    std::cout << "All tests passed" << std::endl;
    return 0;
}
