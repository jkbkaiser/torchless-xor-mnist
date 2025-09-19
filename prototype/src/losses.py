from typing import Tuple

import numpy as np


def binary_cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.float64, np.ndarray]:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels = labels[:, np.newaxis]
    loss = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean()
    grads = probs - labels
    return loss, grads


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
    loss = np.mean(-np.sum(one_hot * logits, axis=-1) + m.squeeze() + log_sum_exp)

    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    grads = softmax - one_hot

    return loss, grads

