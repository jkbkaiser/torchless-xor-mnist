from pathlib import Path

# Directory where images will be saved
IMG_DIR = Path("./figs")
MNIST_DIR = Path("./../data/mnist")

MNIST_TRAIN_IMAGES = MNIST_DIR / "train-images.idx3-ubyte"
MNIST_TRAIN_LABELS = MNIST_DIR / "train-labels.idx1-ubyte"
MNIST_TEST_IMAGES = MNIST_DIR / "t10k-images.idx3-ubyte"
MNIST_TEST_LABELS = MNIST_DIR / "t10k-labels.idx1-ubyte"
