import numpy as np
from numpy.random import default_rng

from src.dataloaders import MNISTDataLoader
from src.losses import cross_entropy_loss
from src.modules import MLP

if __name__ == "__main__":
    num_epochs = 3
    batch_size = 1024
    lr = 0.05
    rng = default_rng(seed=3)

    train_dl = MNISTDataLoader(batch_size, rng, split="TRAIN")
    model = MLP(in_features=784, hidden_features=521, out_features=10, rng=rng)

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

    # batch = next(iter(train_dl))
    # model.forward(batch[0].reshape(batch[0].shape[0], -1))
    # model.layers[0].print()
