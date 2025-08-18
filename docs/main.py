from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

# Directory where images will be saved
IMG_DIR = Path(__file__).parent / "figs"

# Random number generator used for generating random data and
# initializing weights
rng = default_rng()


class Module(ABC):
    """Base class for all neural network modules"""

    @abstractmethod
    def forward(self, x) -> np.ndarray:
        """Forward pass for this module (compute the output)"""
        pass

    @abstractmethod
    def backward(self, grads) -> np.ndarray:
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


def cross_entropy_loss(logits, labels) -> Tuple[np.float64, np.ndarray]:
    """
    Combines softmax computation with cross entropy loss. These two are combined
    because then we can use the log-sum-exp trick that results in efficient and
    stable computations. This function computes both the loss and the gradients.

    The softmax function is defined as follow: $f(x) = exp(x) / sum(exp(x))$
    The cross entropy loss is defined as follow: $f(x, y) = -sum(y * log(x))$

    They can be combined as follows:

    $$
    L       =
    f(x, y) = -sum(y_i * log(softmax(x)))
            = -sum(y_i * log(exp(x_i) / sum(exp(x_j))))
            = -sum(y_i * (log(exp(x_i)) - log(sum(exp(x_j))))
            = -sum(y_i * (x_i - log(sum(exp(x_j))))
            = -sum(y_i * x_i) + sum(y_i * log(sum(exp(x_j))))
            = -sum(y_i * x_i) + sum(y_i * log(sum(exp(x_j) * exp(c) / exp(c))))
            = -sum(y_i * x_i) + sum(y_i * log(exp(c) * sum(exp(x_j - c))))
            = -sum(y_i * x_i) + sum(y_i * (c + log(sum(exp(x_j - c))))
            = -sum(y_i * x_i) + sum(y_i * c) + sum(y_i * log(sum(exp(x_j - c)))
            This works because y uses one-hot encoding
            = -sum(y_i * x_i) + c + log(sum(exp(x_j - c))
    $$

    The derivative w.r.t. the inputs can be computed as follows:

    $$
    (d L)/(d x_i) = (d / d x_i) -sum(y_i * x_i) + c + log(sum(exp(x_j - c))
                  = (d / d x_i) -sum(y_i * x_i) + (d / d x_i) log(sum(exp(x_j - c))
                  = - y_i + 1 / (sum(exp(x_j - c)) * (d / d x_i) sum(exp(x_j - c))
                  = - y_i + 1 / (sum(exp(x_j - c)) * exp(x_i - c)
                  = - y_i + exp(x_i - c) / (sum(exp(x_j - c))
                  = - y_i + softmax(x_i)
    $$

    So the full derivative is a vector: -y + softmax(x)
    """
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(logits.shape[0]), labels] = 1

    m = logits.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    log_sum_exp = np.log(exp_shifted.sum(axis=-1))
    loss = np.sum(-np.sum(one_hot * logits, axis=-1) + m.squeeze() + log_sum_exp)

    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    grads = softmax - one_hot

    return loss, grads


class ReLU(Module):
    """ReLU activation function"""

    def forward(self, x) -> np.ndarray:
        """
        Forward pass for the ReLU activation function. The ReLU function is
        defined as follows: $f(x) = max(x, 0)$
        """
        self.cached_input = x
        return np.maximum(x, 0)

    def backward(self, grads) -> np.ndarray:
        """
        Backward pass for the ReLU activation function. The ReLU function is
        defined as follows: $f(x) = max(x, 0)$. The derivative with respect to
        the inputs is thus 1 if the input is positive and 0 otherwise.
        """
        return grads * (self.cached_input > 0).astype(grads.dtype)

    def update(self, learning_rate):
        """ReLU does not have any trainable parameters"""
        pass

    def zero_grad(self):
        """ReLU does not have any trainable parameters"""
        pass


class LinearLayer(Module):
    """
    Fully connected linear layer

    Weights are initialized using Xavier initialization.
    The bias is initialized to zero.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6 / (in_features + out_features))
        self.w = rng.uniform(-limit, limit, size=(out_features, in_features))
        self.w_grads = np.zeros_like(self.w)

        self.b = np.zeros((out_features,))
        self.b_grads = np.zeros_like(self.b)

    def forward(self, x):
        """
        The fully connected layer is defined as follows using matrix notation:

        $$
        f(x) = x W^T + b
        $$.

        This assumes that the weights are intialized as (out_features,
        in_features).
        """
        self.cached_input = x
        return x @ self.w.T + self.b

    def backward(self, grads):
        """
        Note:
        Notation of the Jacobian: the effect of jth variable with respect to ith
        input is given by w_ij.

        # Update rule for the bias

        Here we assume W has shape (`out_features`, `in_features`) and b has
        shape (`out_features`,).

        First we derive the gradients with respect to the bias b. This can be
        seen as computing the derivative for a function that that takes a
        vector of shape `in_features` and outputs a vector of shape
        `out_features`. So the full jacobian will have the shape (`out_features`,
        `in_features`). Lets look at the derivative of a single output element
        with respect to a single element of b.

        $$
        (d y_k) / (d b_l) = (d) / (d b_l) sum_i W_ik * x_k + b_k
                          = (d) / (d b_l) b_k
                          = kronecker_delta_kl
        $$

        To update the weights of the bias we multiply the Jacobian with the
        gradients of loss w.r.t the output of the function with the gradient of
        the function w.r.t the bias:

        $$
        (d L) / (d b) = (d L) / (d Y) * (d Y) / (d b) = (d L) / (d Y)
        $$

        So this will have the shape of the output of the linear layers which is
        the same as the shape of the bias. The kronecker delta will just sum the
        gradients over the batch.

        # Gradients with respect to the input

        Taking the derivative with respect to the input will have the same shape as
        for the bias. Lets look at the derivative of one element:

        $$
        (d y_k) / (d x_l) = (d) / (d x_l) sum_i W_ik * x_k + b_k
                          = (d) / (d x_l) sum_i W_ik * x_k
                          = sum_i W_ik * kronecker_delta_il
                          = W_lk
        $$

        So the full Jacobian is simply W^T. This can be multiplied with the
        gradiants of the loss w.r.t the output of the function to find the
        gradient of this function w.r.t. toe loss. Because we already store the
        W matrix as W^T this means we can just multiply the grads with the
        stored weight matrix.

        # Gradients with respect to the weights W

        This can be seen as a function that takes in (`in_features`,
        `out_features`) inputs and outputs a vector of shape `out_features`. So
        the full Jacobian would have the shape (`out_features`, `in_features`,
        `out_features`). This is a bit more complex than the previous cases.
        Therefore we look at the derivative of the output with respect to one
        element in the weight matrix W. This will just be a single vector.

        $$
        (d y_o) / (d W_lm) = (d) / (d W_lm) sum_k W_ok * x_k + b_k
                           = sum_k x_k * (d) / (d W_lm) W_ok
                           = sum_k x_k * kronecker_delta_lo * kronecker_delta_mk
                           = x_m * kronecker_delta_lo
        $$

        Now lets compute:

        $$
        (d L) / (d W_lm) = (d L) / (d y) * (d y) / (d W_lm)
                         = z * (d y) / (d W_lm)
                         = sum_o z_o * (d y_o) / (d W_lm)
                         = sum_o z_o * (d y_o) / (d W_lm)
                         = sum_o z_o * x_m * kronecker_delta_lo
                         = z_l * x_m
        $$

        This means that the overall gradient can be written as the outproduct:

        $$
        (d L) / (d W) = z^T x^T
        $$
        """
        self.b_grads = grads.sum(axis=0)
        # Because the inputs are a batch they are already transposed.
        self.w_grads = grads.T @ self.cached_input
        return grads @ self.w

    def update(self, learning_rate: float):
        """
        Updates the weights of the linear layer in the opposite directions
        of the gradients.
        """
        self.w -= learning_rate * self.w_grads
        self.b -= learning_rate * self.b_grads

    def zero_grad(self):
        """Resets the gradients for the weights and bias."""
        self.w_grads = np.zeros_like(self.w)
        self.b_grads = np.zeros_like(self.b)


class MLP(Module):
    """ Multi-layer perceptron """

    def __init__(self, in_features, hidden_features, out_features):
        """ Initialize the MLP with the given input and output dimensions """
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.layers = [
            LinearLayer(in_features=in_features, out_features=hidden_features),
            ReLU(),
            LinearLayer(in_features=hidden_features, out_features=out_features),
        ]

    def forward(self, x):
        """Passes the input through the MLP layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grads):
        """Passes the gradients backwards through the MLP layers."""
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads

    def update(self, learning_rate: float):
        """Updates the weights of the MLP layers."""
        for layer in self.layers:
            layer.update(learning_rate)

    def zero_grad(self):
        """Resets the gradients of the MLP layers."""
        for layer in self.layers:
            layer.zero_grad()


class DataLoader:
    """Simple generator the generates batches of XOR data samples."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __next__(self):
        a = (rng.uniform(0, 1, size=(self.batch_size,)) > 0.5).astype(np.int8)
        b = (rng.uniform(0, 1, size=(self.batch_size,)) > 0.5).astype(np.int8)
        return np.stack((a, b), axis=1), a^b


def visualize_decision_boundary(model, figname):
    a = np.linspace(0, 1, 100)
    b = np.linspace(0, 1, 100)
    x, y = np.meshgrid(a, b)
    grid_points = np.stack([x.ravel(), y.ravel()], axis=-1)
    outputs = np.array([model.forward(p) for p in grid_points])  # shape: (200*200, out_features)
    preds = np.argmax(outputs, axis=1)
    z = preds.reshape(x.shape)

    _, ax = plt.subplots(figsize=(6, 6))

    ax.contourf(x, y, z, cmap="coolwarm", alpha=0.5)

    if outputs.shape[1] == 2:  # only makes sense for binary classification
        boundary = (outputs[:, 0] - outputs[:, 1]).reshape(x.shape)
        ax.contour(x, y, boundary, levels=[0], colors="black", linewidths=2)

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title("Decision Boundary")

    plt.savefig(figname)


if __name__ == "__main__":
    if not IMG_DIR.exists():
        IMG_DIR.mkdir(parents=True)

    decision_boundary_visualization_path = IMG_DIR / "decision_boundary"

    if not decision_boundary_visualization_path.exists():
        decision_boundary_visualization_path.mkdir(parents=True)

    # Hyperparameters
    num_epochs = 10
    batches_per_epoch = 10
    batch_size = 16
    lr = 0.01

    dl = DataLoader(batch_size)
    model = MLP(in_features=2, hidden_features=8, out_features=2)

    losses = []
    accs = []

    for epoch in range(num_epochs):
        avg_loss = 0
        avg_acc = 0

        for _ in range(batches_per_epoch):
            x, y = next(dl)
            logits = model.forward(x)

            pred = logits.argmax(axis=-1)
            acc = ((pred == y).sum() / batch_size).item()

            loss, grads = cross_entropy_loss(logits, y)

            model.backward(grads)
            model.update(learning_rate=lr)
            model.zero_grad()

            avg_loss += loss.item()
            avg_acc += acc

        avg_loss /= batches_per_epoch
        avg_acc /= batches_per_epoch

        print(f"Epoch {epoch} {avg_acc=} {avg_loss=}")

        losses.append(avg_loss)
        accs.append(avg_acc)

        visualize_decision_boundary(
            model,
            figname=decision_boundary_visualization_path / f"{epoch}.png"
        )

    # Losses
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(IMG_DIR / "losses.png")

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(accs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    plt.savefig("figs/accs.png")
