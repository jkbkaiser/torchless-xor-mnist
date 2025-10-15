#include <iomanip>

#include "tensor.h"
#include "losses.h"
#include "nn.h"
#include "dataloaders.h"


double is_correct(const Tensor& output, const Tensor& label) {
  double max = output.at({0});
  int max_idx = 0;

  for (int i = 1; i < output.shape[0]; ++i) {
    if (output.at({i}) > max) {
      max = output.at({i});
      max_idx = i;
    }
  }

  return label.at({max_idx}) == 1.0 ? 1.0 : 0.0;
}


int main() {
  int num_epochs = 10;
  int batch_size = 64;
  int batches_per_epoch = 10;
  double learning_rate = 0.5;

  MNISTDataLoader dl(batch_size, TRAIN);
  MLP model(2, 4, 2);
  CrossEntropyLoss criterion{};

  int iter = 0;
  int epoch = 0;

  std::vector<double> losses;
  std::vector<double> accs;

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    int avg_loss = 0;
    int avg_acc = 0;

    for (int i = 0; i < batches_per_epoch; ++i) {
      auto [x, y] = dl.next();
      Tensor logits = model.forward(x);
  //
  //     auto [loss, loss_grads] = criterion(logits, y);
  //
  //     model.backward(loss_grads);
  //     model.update(learning_rate);
  //     model.zero_grad();
  //
  //     avg_acc += is_correct(logits, y);
  //     avg_loss += loss;
    }
  //
  //   avg_loss /= batches_per_epoch;
  //   avg_acc /= batches_per_epoch;
  //
  //   losses.push_back(avg_loss);
  //   accs.push_back(avg_acc);
  //
  //   std::cout << "Epoch " << std::setw(3) << epoch <<  " avg_acc=" << std::setprecision(4) << std::setw(6) << std::left << avg_acc << std::setprecision(4) << std::setw(6) << std::left << " avg_loss=" << avg_loss << std::endl;
  }
}
