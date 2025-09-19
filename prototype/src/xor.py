import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from src.constants import IMG_DIR
from src.dataloaders import XORDataLoader
from src.losses import binary_cross_entropy_loss
from src.modules import MLP


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    if not IMG_DIR.exists():
        IMG_DIR.mkdir(parents=True)

    noise_std = 0.1
    num_epochs = 1000
    batches_per_epoch = 10
    batch_size = 64
    lr = 0.05
    rng = default_rng(seed=3)

    dl = XORDataLoader(batch_size, noise_std, rng)
    model = MLP(in_features=2, hidden_features=4, out_features=1, rng=rng)

    losses = []
    accs = []

    for epoch in range(num_epochs):
        avg_loss = 0
        avg_acc = 0

        for _ in range(batches_per_epoch):
            x, y = next(dl)
            logits = model.forward(x)

            probs = sigmoid(logits)
            preds = (probs >= 0.5).astype(int)
            acc = ((preds.squeeze() == y).sum() / batch_size).item()

            loss, grads = binary_cross_entropy_loss(probs, y)

            model.backward(grads)
            model.update(learning_rate=lr)
            model.zero_grad()

            avg_loss += loss.item()
            avg_acc += acc

        avg_loss /= batches_per_epoch
        avg_acc /= batches_per_epoch

        losses.append(avg_loss)
        accs.append(avg_acc)

        print(f"Epoch {epoch} {avg_acc=} {avg_loss=}")

        if epoch > 0 and abs(losses[-1] - losses[-2]) < 1e-5:
            break

    # Losses
    fig, ax = plt.subplots(figsize=(8, 8))

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.plot(losses, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(IMG_DIR / "xor_losses.png")

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 8))

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.plot(accs, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    plt.savefig(IMG_DIR / "xor_accs.png")

    # Decision boundaries
    coords = np.linspace(0, 1, 100)
    x, y = np.meshgrid(coords, coords)
    inputs = np.stack([x.ravel(), y.ravel()], axis=1)
    probs = sigmoid(model.forward(inputs))
    preds = (probs >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(8, 8))

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    mask = ((inputs[:, 0] < 1) & (inputs[:, 1] < 1))
    zero_preds = inputs[(preds==0).squeeze() & mask]
    one_preds = inputs[(preds==1).squeeze() & mask]

    ax.scatter(zero_preds[:, 0], zero_preds[:, 1], facecolors='none', edgecolors='black', linewidths=0.5, label="0 label")
    ax.scatter(one_preds[:, 0], one_preds[:, 1], marker="x", color="black", linewidths=0.5 , label="1 label")

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    plt.savefig(IMG_DIR / "xor_decision_boundaries.png")
