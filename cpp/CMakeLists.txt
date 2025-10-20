cmake_minimum_required(VERSION 3.18)
project(torchless LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wpedantic")

# Build outputs go to build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# XOR executable
add_executable(xor src/xor.cpp src/tensor.cpp)

# MNIST executable
add_executable(mnist src/mnist.cpp src/tensor.cpp)

# Tests executable
# add_executable(tests src/tests.cpp src/tensor.cpp)
