import numpy as np
from numpy.random import default_rng

from src.dataloaders import MNISTDataLoader
from src.losses import cross_entropy_loss
from src.modules import MLP

if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 16
    lr = 0.05
    rng = default_rng(seed=3)

    train_dl = MNISTDataLoader(batch_size, rng, split="TRAIN")
    model = MLP(in_features=784, hidden_features=1024, out_features=10, rng=rng)

    losses = []
    accs = []

    for epoch in range(num_epochs):
        avg_loss = 0
        avg_acc = 0

        print("Epoch")

        for batch in train_dl:
            x, y = batch
            x_flat = x.reshape(x.shape[0], -1)

            logits = model.forward(x_flat)
            loss, grads = cross_entropy_loss(logits, y)

            model.backward(grads)
            model.update(learning_rate=lr)
            model.zero_grad()

            avg_loss += loss.item()
            avg_acc += np.mean(np.argmax(logits, axis=1) == y)

        avg_loss /= len(train_dl)
        avg_acc /= len(train_dl)

        losses.append(avg_loss)
        accs.append(avg_acc)

        print(f"Epoch {epoch} {avg_acc=} {avg_loss=}")

        # if epoch > 0 and abs(losses[-1] - losses[-2]) < 1e-5:
        #     break
