from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):
    """Base class for all neural network modules"""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for this module (compute the output)"""
        pass

    @abstractmethod
    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass for this module (compute the gradients)"""
        pass

    @abstractmethod
    def update(self, learning_rate: float):
        """Update the weights of this module"""
        pass

    @abstractmethod
    def zero_grad(self):
        """Reset the gradients of this module"""
        pass


class LinearLayer(Module):
    """
    Fully connected linear layer

    Weights are initialized using Xavier initialization.
    The bias is initialized to zero.
    """

    def __init__(self, in_features, out_features, rng: np.random.Generator):
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6 / (in_features + out_features))
        self.w = rng.uniform(-limit, limit, size=(out_features, in_features))
        self.w_grads = np.zeros_like(self.w)

        self.b = np.zeros((out_features,))
        self.b_grads = np.zeros_like(self.b)

    def forward(self, x):
        self.cached_input = x
        return x @ self.w.T + self.b

    def backward(self, grads):
        self.b_grads = grads.sum(axis=0) / grads.shape[0]
        self.w_grads = grads.T @ self.cached_input / grads.shape[0]
        return grads @ self.w

    def update(self, learning_rate: float):
        self.w -= learning_rate * self.w_grads
        self.b -= learning_rate * self.b_grads

    def zero_grad(self):
        self.w_grads = np.zeros_like(self.w)
        self.b_grads = np.zeros_like(self.b)


class Tanh(Module):
    """Tanh activation function"""

    def forward(self, x):
        self.cached = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.cached

    def backward(self, grads):
        return grads * (1 - self.cached**2)

    def update(self, learning_rate):
        pass

    def zero_grad(self):
        pass


class MLP(Module):
    """Multi-layer perceptron"""

    def __init__(self, in_features, hidden_features, out_features, rng: np.random.Generator):
        self.layers = [
            LinearLayer(in_features=in_features, out_features=hidden_features, rng=rng),
            Tanh(),
            LinearLayer(in_features=hidden_features, out_features=out_features, rng=rng),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grads):
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads

    def update(self, learning_rate: float):
        for layer in self.layers:
            layer.update(learning_rate)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
