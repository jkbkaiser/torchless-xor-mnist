from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.constants import (MNIST_TEST_IMAGES, MNIST_TEST_LABELS,
                           MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS)


class XORDataLoader:
    """Simple generator that generates batches of XOR data samples with noise."""


    def __init__(self, batch_size: int, noise_std: float, rng: np.random.Generator):
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.rng = rng

    def __next__(self):
        a = self.rng.integers(0, 2, size=(self.batch_size,))
        b = self.rng.integers(0, 2, size=(self.batch_size,))

        y = a^b
        x = np.stack((a, b), axis=1)
        noise = self.rng.uniform(0, self.noise_std, size=x.shape)

        return x + noise, y


class MNISTDataLoader:
    """Simple generator that generates batches of MNIST data samples."""

    def __init__(
        self,
        batch_size: int,
        rng: np.random.Generator,
        split: Literal["TRAIN"] | Literal["TEST"]
    ):
        self.batch_size = batch_size
        self.rng = rng

        if split == "TRAIN":
            self.images = np.fromfile(MNIST_TRAIN_IMAGES, dtype=np.uint8, offset=16)
            self.labels = np.fromfile(MNIST_TRAIN_LABELS, dtype=np.uint8, offset=8)
        elif split == "TEST":
            self.images = np.fromfile(MNIST_TEST_IMAGES, dtype=np.uint8, offset=16)
            self.labels = np.fromfile(MNIST_TEST_LABELS, dtype=np.uint8, offset=8)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.images = self.images.reshape((-1, 28, 28))
        self.labels = self.labels.reshape((-1,))
        self.num_samples = self.images.shape[0]

        # Initialize indices
        self.indices = np.arange(self.num_samples)
        self.ptr = 0

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        self.rng.shuffle(self.indices)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.num_samples:
            raise StopIteration

        end = min(self.ptr + self.batch_size, self.num_samples)
        batch_idx = self.indices[self.ptr:end]

        x = self.images[batch_idx]
        y = self.labels[batch_idx]

        self.ptr = end
        return x, y

    def __len__(self):
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    dl = MNISTDataLoader(batch_size=16, rng=np.random.default_rng(seed=3), split="TRAIN")
    samples, _ = next(iter(dl))

    print(samples.shape)

    fig, ax = plt.subplots(4, 4,figsize=(8, 8))
    fig.patch.set_alpha(0.0)

    for i, sample in enumerate(samples):
        ax[i // 4, i % 4].imshow(sample, cmap="gray")
        ax[i // 4, i % 4].axis("off")
        ax[i // 4, i % 4].set_xticks([])
        ax[i // 4, i % 4].set_yticks([])

    fig.savefig("figs/mnist_train_samples.png")
