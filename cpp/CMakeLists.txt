cmake_minimum_required(VERSION 3.18)
project(torchless LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wpedantic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build outputs go to build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Library - exposes tensor, nn, dataloaders, losses
add_library(torchless STATIC
    # Tensor sources

    # Wrapper implementations
    src/tensor/tensor.cpp

    # Base implementations
    src/tensor/base/tensor_factories.cpp

    # CPU implementations
    src/tensor/cpu/tensor_factories.cpp
    src/tensor/cpu/tensor_index.cpp
    src/tensor/cpu/tensor_io.cu
    # src/tensor/cpu/tensor_ops.cpp
    # src/tensor/cpu/tensor_linalg.cpp
    # src/tensor/cpu/tensor_reduction.cpp
    # src/tensor/cpu/tensor_shape.cpp

    # GPU implementations
    src/tensor/gpu/tensor_factories.cu

    # NN Sources
    # src/nn/modules.cpp
    # src/nn/losses.cpp

    # Data sources
    # src/data.cpp
)

target_include_directories(torchless
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# XOR executable
add_executable(xor examples/xor.cpp)
target_link_libraries(xor PRIVATE torchless)

# MNIST executable
add_executable(mnist examples/mnist.cpp)
target_link_libraries(mnist PRIVATE torchless)

# Tests executable
add_executable(tests
    tests/main.cpp
    tests/tensor.cpp
)
target_link_libraries(tests PRIVATE torchless)
