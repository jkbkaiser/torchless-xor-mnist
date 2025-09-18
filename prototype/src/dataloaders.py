import numpy as np


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
