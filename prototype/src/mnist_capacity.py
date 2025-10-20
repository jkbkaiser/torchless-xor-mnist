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

    num_epochs = 500
    batch_size = 1024
    lr = 0.05
    seed = 3
    convergence_threshold = 1e-5

    # Test different numbers of hidden neurons
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    test_accuracies = []

    for hidden_features in hidden_sizes:
        print(f"\n{'='*50}")
        print(f"Training with {hidden_features} hidden neurons")
        print(f"{'='*50}")

        rng = default_rng(seed=seed)
        train_dl = MNISTDataLoader(batch_size, rng, split="TRAIN")
        model = MLP(in_features=784, hidden_features=hidden_features, out_features=10, rng=rng)

        # Training loop
        losses = []
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

            if epoch % 10 == 0:
                print(f"Epoch {epoch:>3} {avg_acc=:.4f} {avg_loss=:.4f}")

            # Early stopping if converged
            if epoch > 10 and abs(losses[-1] - losses[-2]) < convergence_threshold:
                print(f"Converged at epoch {epoch}")
                break

        # Evaluate on test set
        test_dl = MNISTDataLoader(batch_size, rng, split="TEST")
        test_acc = 0

        for batch in test_dl:
            x, y = batch
            x_flat = x.reshape(x.shape[0], -1)
            logits = model.forward(x_flat)
            test_acc += np.sum(np.argmax(logits, axis=1) == y)

        test_acc = test_acc / len(test_dl)
        test_accuracies.append(test_acc)
        print(f"Test accuracy: {test_acc:.2f}")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 8))

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.plot(hidden_sizes, test_accuracies, 'o-', color='black', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Hidden Neurons', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(IMG_DIR / "capacity_vs_accuracy.png", dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*50}")
    print("Results Summary:")
    print(f"{'='*50}")
    for h, acc in zip(hidden_sizes, test_accuracies):
        print(f"{h:>3} neurons: {acc:.2f} accuracy")
