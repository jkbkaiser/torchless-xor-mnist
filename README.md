# Torchless XOR/MNIST

This project contains an implementation of the Multi-Layer Perceptron from
scratch with CUDA support. The model is applied to the XOR problem and the MNIST
dataset. A detailed discussion of the necessary derivations and implementation
can be found on my [blog](https://jakobkaiser.com/blog/torchless-xor-mnist/).

## Dependencies

The prototype was developed using Python 3.13.5 and depends on Numpy version
2.3.2 and Matplotlib version 3.10.5 for visualisations. These can be installed
using [uv](https://github.com/astral-sh/uv) by runnning the command `uv sync`
from the python directory.

The C++ implementation developed using C++ version 23. It has no dependencies
except CUDA when you want to use the GPU.

## Structure

## Usage

To run the prototype, go into the `python` directory, download dependencies, and
you are ready to run the code.

```bash
cd python
uv sync

# Now you can execute the training scripts for XOR and MNIST
python -m src.xor
python -m src.mnist
```

To compile the c++ code, you can the following commands:

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

You can now run the following binaries from the build directory:

```bash
./xor
./mnist
./test
```
