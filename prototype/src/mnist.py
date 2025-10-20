import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from src.constants import IMG_DIR
from src.dataloaders import MNISTDataLoader
from src.losses import cross_entropy_loss
from src.modules import MLP

if __name__ == "__main__":
    if not IMG_DIR.exists():
        IMG_DIR.mkdir(parents=True)

    num_epochs = 25
    batch_size = 1024
    lr = 0.05
    rng = default_rng(seed=3)

    train_dl = MNISTDataLoader(batch_size, rng, split="TRAIN")
    model = MLP(in_features=784, hidden_features=10, out_features=10, rng=rng)

    losses = []
    accs = []

    for epoch in range(num_epochs):
        avg_loss = 0
        avg_acc = 0

        for batch in train_dl:
            x, y = batch
            x_flat = x.reshape(x.shape[0], -1)

            logits = model.forward(x_flat)
            loss, grads = cross_entropy_loss(logits, y)

            model.backward(grads)
            model.update(learning_rate=lr)
            model.zero_grad()

            avg_loss += loss.item()
            avg_acc += np.sum(np.argmax(logits, axis=1) == y).item()

        avg_loss /= len(train_dl)
        avg_acc /= len(train_dl)

        losses.append(avg_loss)
        accs.append(avg_acc)

        print(f"Epoch {epoch:>3} {avg_acc=:.4f} {avg_loss=:.4f}")

        if epoch > 0 and abs(losses[-1] - losses[-2]) < 1e-4:
            break


    train_dl = MNISTDataLoader(batch_size, rng, split="TEST")
    avg_acc = 0

    for batch in train_dl:
        x, y = batch
        x_flat = x.reshape(x.shape[0], -1)
        logits = model.forward(x_flat)
        avg_acc += np.sum(np.argmax(logits, axis=1) == y)

    print(f"Test accuracy: {avg_acc / len(train_dl):.2f}")

    x, y = next(iter(train_dl))
    first_sample = x[0]
    first_label = y[0]

    logits = model.forward(first_sample.reshape(1, 784))
    loss, grads = cross_entropy_loss(logits, first_label.reshape(1, 1))

    model.zero_grad()
    model.backward(grads)

    first_layer = model.layers[0]


    # Visualize neurons
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_alpha(0.0)

    # Find global min/max for consistent scaling
    all_weights = [first_layer.w[i] for i in range(9)]
    vmin = min(w.min() for w in all_weights)
    vmax = max(w.max() for w in all_weights)

    for i in range(9):
        w = first_layer.w[i]
        w = np.reshape(w, (28, 28))
        ax = axes[i // 3, i % 3]
        ax.patch.set_alpha(0.0)
        im = ax.imshow(w, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(IMG_DIR / f"weights_grid.png", dpi=300, transparent=True, bbox_inches='tight')


    # Compute average activation per digit class for each neuron
    train_dl_full = MNISTDataLoader(batch_size, rng, split="TRAIN")
    neuron_activations_by_digit = {i: {digit: [] for digit in range(10)} for i in range(10)}

    for batch in train_dl_full:
        x, y = batch
        x_flat = x.reshape(x.shape[0], -1)

        hidden_output = first_layer.forward(x_flat)
        activations = model.layers[1].forward(hidden_output)  # After ReLU

        for digit in range(10):
            mask = (y == digit)
            if mask.sum() > 0:
                digit_activations = activations[mask]
                for neuron_idx in range(10):
                    neuron_activations_by_digit[neuron_idx][digit].extend(
                        digit_activations[:, neuron_idx].tolist()
                    )

    # Compute mean activation per neuron per digit
    mean_activations = np.zeros((10, 10))
    for neuron_idx in range(10):
        for digit in range(10):
            mean_activations[neuron_idx, digit] = np.mean(
                neuron_activations_by_digit[neuron_idx][digit]
            )

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.patch.set_alpha(0.0)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        ax.bar(range(10), mean_activations[i], color='black', alpha=0.7)
        ax.set_xlabel('Digit', fontsize=10)
        ax.set_ylabel('Mean Activation', fontsize=10)
        ax.set_title(f'Neuron {i}', fontsize=12)
        ax.set_xticks(range(10))
        ax.set_ylim(0, mean_activations.max() * 1.1)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(IMG_DIR / f"neuron_digit_activations.png", dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
