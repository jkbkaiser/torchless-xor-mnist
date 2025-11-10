// #include "torchless/tensor/tensor_cpu.h"
// #include <random>

// #include <torchless/data.h>
// #include <torchless/nn.h>
// #include <cfloat>
#include <torchless/tensor.h>
#include <torchless/utils.h>

// Tensor sigmoid(const Tensor &x) { return 1 / (1 + (-x).exp()); }

int main() {
    // int seed = 0;
    // int num_epochs = 100;
    // int batch_size = 64;
    // double lr = 0.05;
    // double noise_std = 0.1;

    //
    // CPUTensor t(shape);

    Tensor t({1, 2});

    std::cout << t << std::endl;

    // std::mt19937 rng(seed);
    //
    // XORDataset ds(batch_size * 10, noise_std, rng);
    // Dataloader dl(&ds, batch_size, rng);
    //
    // MLP model(2, 10, 1, rng);
    // BinaryCrossEntropyLoss criterion{};
    //
    // std::vector<double> losses;
    // std::vector<double> accs;
    //
    // for (int epoch = 0; epoch < num_epochs; ++epoch) {
    //     double avg_loss = 0;
    //     double avg_acc = 0;
    //
    //     for (auto [x, y] : dl) {
    //         Tensor logits = model.forward(x);
    //         Tensor probs = sigmoid(logits);
    //         Tensor preds = probs.map([](double x) { return x > 0.5 ? 1.0 : 0.0; });
    //         double acc = (preds == y).sum().item();
    //         auto [loss, grads] = criterion(probs, y);
    //
    //         model.zero_grad();
    //         model.backward(grads);
    //         model.update(lr);
    //
    //         avg_loss += loss;
    //         avg_acc += acc;
    //     }
    //
    //     avg_loss /= ds.size();
    //     avg_acc /= ds.size();
    //
    //     losses.push_back(avg_loss);
    //     accs.push_back(avg_acc);
    //
    //     log_epoch(epoch, avg_loss, avg_acc);
    // }
}
