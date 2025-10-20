from typing import Tuple

import numpy as np


def binary_cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.float64, np.ndarray]:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels = labels[:, np.newaxis]
    loss = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean()
    grads = probs - labels
    return loss, grads


def cross_entropy_loss(logits, labels) -> Tuple[np.float64, np.ndarray]:
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(logits.shape[0]), labels] = 1

    m = logits.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    log_sum_exp = np.log(exp_shifted.sum(axis=-1))
    loss = np.sum(-np.sum(one_hot * logits, axis=-1) + m.squeeze() + log_sum_exp)

    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    grads = softmax - one_hot

    return loss, grads

