#include <iomanip>

#include "losses.h"
#include "nn.h"
#include "dataloaders.h"

// Tensor sigmoid(const Tensor& x) {
//   return 1 / (1 + (-x).exp());
// }

int main() {
    int num_epochs = 100;
    int batch_size = 64;
    int batches_per_epoch = 10;
    double lr = 0.05;
    double noise_std = 0.1;

    XORDataSet ds(1000, noise_std);
    Dataloader dl(&ds, batch_size);

    // XORDataLoader dl(batch_size, noise_std);
    // MLP model(2, 10, 1);
    // BinaryCrossEntropyLoss criterion{};
    //
    // double avg_accuracy = 0.0;
    // double avg_loss = 0.0;
    //
    // int iter = 0;
    // int epoch = 0;
    //
    // std::vector<double> losses;
    // std::vector<double> accs;
    //
    // for (int epoch = 0; epoch < num_epochs; ++epoch) {
    //     double avg_loss = 0;
    //     double avg_acc = 0;
    //
    //     for (int i = 0; i < batches_per_epoch; ++i) {
    //         auto [x, y] = dl.next();
    //         Tensor logits = model.forward(x);
    //
    //         Tensor probs = sigmoid(logits);
    //         Tensor preds = probs.map([](double x) { return x >= 0.5 ? 1.0 : 0.0; });
    //         double acc = (preds.squeeze() == y).sum().item() / batch_size;
    //
    //         auto [loss, grads] = criterion(probs, y);
    //
    //         model.backward(grads);
    //         model.update(lr);
    //         model.zero_grad();
    //
    //         avg_loss += loss;
    //         avg_acc += acc;
    //     }
    //
    //     avg_loss /= batches_per_epoch;
    //     avg_acc /= batches_per_epoch;
    //
    //     losses.push_back(avg_loss);
    //     accs.push_back(avg_acc);
    //
    //     std::cout << "Epoch " << std::setw(3) << epoch <<  " avg_acc=" << std::setprecision(4) << std::setw(6) << std::left << avg_acc << std::setprecision(4) << std::setw(6) << std::left << " avg_loss=" << avg_loss << std::endl;
    // }
}

