# Torchless XOR/MNIST

A from-scratch neural network library implemented in C++ with CUDA support. This project demonstrates a Multi-Layer Perceptron applied to the XOR problem and MNIST digit classification, built without external ML frameworks.

A detailed discussion of the mathematical derivations and implementation can be found on my [blog](https://jakobkaiser.com/blog/torchless-xor-mnist/).

## Features

- **Pure C++23** implementation with no dependencies (except CUDA for GPU support)
- **Custom tensor library** with broadcasting and matrix operations
- **Manual backpropagation**: Hand-coded gradient computation for all operations
- **Neural network modules**: Linear layers, ReLU activation, MLP
- **Loss functions**: Cross-entropy and binary cross-entropy
- **Python prototype** for validation

## Project Structure

```
torchless-xor/
├── cpp/                    # C++ implementation (primary)
├── python/                 # Python reference implementation
└── data/                   # Datasets (MNIST)
```

## Quick Start

### C++ Implementation

**Requirements:**
- C++23 compatible compiler (GCC 12+, Clang 15+)
- CMake 3.18+
- CUDA Toolkit (optional, for GPU support)

**Build and run:**
```bash
cd cpp
mkdir build && cd build
cmake ..
make

# Run examples
./xor
./mnist

# Run tests
./tests
```

**Build with CUDA:**
```bash
cmake -DUSE_CUDA=ON ..
make
```

### Python Prototype

**Requirements:**
- Python 3.13+
- Dependencies managed via [uv](https://github.com/astral-sh/uv)

**Run:**
```bash
cd prototype
uv sync

# Execute training scripts
python -m src.xor
python -m src.mnist
```

## Examples

### XOR Problem
Trains a simple MLP to learn the XOR function with noise tolerance.

### MNIST Classification
Trains a neural network on the MNIST handwritten digit dataset.

## Blog

For a detailed walkthrough of the implementation, including mathematical derivations and design decisions, visit: [jakobkaiser.com/blog/torchless-xor-mnist](https://jakobkaiser.com/blog/torchless-xor-mnist/)
