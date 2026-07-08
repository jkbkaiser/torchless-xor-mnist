# Torchless XOR/MNIST

A from-scratch neural network library implemented in pure NumPy — no ML frameworks. This project demonstrates a Multi-Layer Perceptron with hand-coded backpropagation applied to the XOR problem and MNIST digit classification.

A detailed discussion of the mathematical derivations and implementation can be found on my [blog](https://jakobkaiser.com/blog/torchless-xor-mnist/).

## Features

- **Manual backpropagation**: hand-coded gradient computation for all operations
- **Neural network modules**: linear layers, ReLU/Tanh activations, MLP
- **Loss functions**: cross-entropy and binary cross-entropy
- **Data loaders**: noisy XOR generator and MNIST (IDX format) loader
- **Visualizations**: training curves, decision boundaries, learned weights, neuron activations

> A C++/CUDA implementation of the same ideas lives on the [`cpp` branch](../../tree/cpp).

## Project Structure

```
torchless-xor/
├── python/
│   └── src/
│       ├── modules.py      # Linear, ReLU, Tanh, MLP with forward/backward
│       ├── losses.py       # Cross-entropy and binary cross-entropy
│       ├── dataloaders.py  # XOR and MNIST data loaders
│       ├── xor.py          # XOR training script
│       ├── mnist.py        # MNIST training script
│       └── mnist_capacity.py  # Model capacity vs. accuracy experiment
└── data/                   # Datasets (MNIST, not included — see below)
```

## Quick Start

**Requirements:**
- Python 3.13+
- Dependencies managed via [uv](https://github.com/astral-sh/uv)

**Setup:**
```bash
cd python
uv sync
```

**Run XOR:**
```bash
uv run python -m src.xor
```

**Run MNIST:**

Download the MNIST dataset (IDX format, e.g. from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)) and place the files in `data/mnist/`:

```
data/mnist/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

Then:
```bash
uv run python -m src.mnist
uv run python -m src.mnist_capacity
```

Generated plots are written to `python/figs/`.

## Examples

### XOR Problem
Trains a simple MLP to learn the XOR function with noise tolerance, and plots the learned decision boundary.

### MNIST Classification
Trains a neural network on the MNIST handwritten digit dataset, visualizes the learned first-layer weights, and explores how test accuracy scales with hidden-layer size.

## Blog

For a detailed walkthrough of the implementation, including mathematical derivations and design decisions, visit: [jakobkaiser.com/blog/torchless-xor-mnist](https://jakobkaiser.com/blog/torchless-xor-mnist/)
