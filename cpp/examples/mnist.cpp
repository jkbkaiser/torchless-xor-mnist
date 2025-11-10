// #include <string>
//
// #include <torchless/data.h>
// #include <torchless/nn.h>
// #include <torchless/tensor.h>
// #include <torchless/utils.h>
//
// // double is_correct(const Tensor &logits, const Tensor &labels) {
// //     const auto &shape = logits.shape;
// //     size_t batch_size = shape[0];
// //     size_t num_classes = shape[1];
// //     Tensor flattened_labels = labels.squeeze();
// //
// //     double correct = 0.0;
// //
// //     for (size_t b = 0; b < batch_size; ++b) {
// //         double max_val = logits.at({b, 0});
// //         int pred_idx = 0;
// //         for (size_t c = 1; c < num_classes; ++c) {
// //             double val = logits.at({b, c});
// //             if (val > max_val) {
// //                 max_val = val;
// //                 pred_idx = c;
// //             }
// //         }
// //
// //         int true_idx = static_cast<int>(flattened_labels.at({b}));
// //
// //         if (pred_idx == true_idx) {
// //             correct += 1.0;
// //         }
// //     }
// //
// //     return correct;
// // }
//
int main() {
    //     // int seed = 0;
    //     // int num_epochs = 10;
    //     // int batch_size = 64;
    //     // double lr = 0.05;
    //
    //     std::vector<size_t> shape = {1, 2};
    //     auto t = Tensor<Device::CPU>::empty(shape);
    //
    //     // std::string mnist_dir = "./../../data/mnist";
    //     //
    //     // std::mt19937 rng(seed);
    //     //
    //     // MNISTDataset train_ds(TRAIN, mnist_dir);
    //     // Dataloader dl(&train_ds, batch_size, rng);
    //     //
    //     // MLP model(784, 2, 10, rng);
    //     // CrossEntropyLoss criterion{};
    //     //
    //     // std::vector<double> losses;
    //     // std::vector<double> accs;
    //     //
    //     // for (int epoch = 0; epoch < num_epochs; ++epoch) {
    //     //     double avg_loss = 0;
    //     //     double avg_acc = 0;
    //     //
    //     //     for (auto [x, y] : dl) {
    //     //         x = x / 255.0;
    //     //         Tensor logits = model.forward(x);
    //     //
    //     //         auto [loss, loss_grads] = criterion(logits, y);
    //     //
    //     //         model.zero_grad();
    //     //         model.backward(loss_grads);
    //     //         model.update(lr);
    //     //
    //     //         double acc = is_correct(logits, y);
    //     //         avg_acc += acc;
    //     //         avg_loss += loss;
    //     //     }
    //     //
    //     //     avg_loss /= train_ds.size();
    //     //     avg_acc /= train_ds.size();
    //     //
    //     //     losses.push_back(avg_loss);
    //     //     accs.push_back(avg_acc);
    //     //
    //     //     log_epoch(epoch, avg_loss, avg_acc);
    //     // }
}
