#ifndef LOSSES_H
#define LOSSES_H

#include <cmath>
#include <utility>

#include "tensor.h"

class CrossEntropyLoss {
  public:
    std::pair<double, Tensor> operator()(const Tensor &logits, const Tensor &labels) const {
        int batch_size = logits.shape[0];
        int num_classes = logits.shape[1];
        Tensor flattened_labels = labels.squeeze();

        double total_loss = 0.0;
        Tensor grad = Tensor::zeros({batch_size, num_classes});

        // Compute loss and gradients per sample
        for (int i = 0; i < batch_size; ++i) {
            int true_class = static_cast<int>(flattened_labels.at({i}));

            // Find max logit for numerical stability
            double max_logit = logits.at({i, 0});
            for (int c = 1; c < num_classes; ++c) {
                max_logit = std::max(max_logit, logits.at({i, c}));
            }

            // Compute log_sum_exp for this sample
            double sum_exp = 0.0;
            for (int c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(logits.at({i, c}) - max_logit);
            }
            double log_sum_exp = std::log(sum_exp);

            // Loss for this sample: -log(p_true_class)
            double sample_loss = -(logits.at({i, true_class}) - max_logit - log_sum_exp);
            total_loss += sample_loss;

            // Gradient: softmax - one_hot
            for (int c = 0; c < num_classes; ++c) {
                double softmax_c = std::exp(logits.at({i, c}) - max_logit - log_sum_exp);
                double one_hot_c = (c == true_class) ? 1.0 : 0.0;
                grad.at({i, c}) = (softmax_c - one_hot_c) / batch_size;
            }
        }

        return {total_loss / batch_size, grad};
    }
};

// class CrossEntropyLoss {
//   public:
//     std::pair<double, Tensor> operator()(const Tensor &logits, const Tensor &labels) const {
//         int batch_size = logits.shape[0];
//         int num_classes = logits.shape[1];
//         Tensor flattened_labels = labels.squeeze();
//
//         Tensor one_hot = Tensor::filled({batch_size, num_classes}, 0.0);
//         for (int i = 0; i < batch_size; ++i) {
//             int cls = static_cast<int>(flattened_labels.at({i}));
//             one_hot.at({i, cls}) = 1.0;
//         }
//
//         double m = logits.max().item();
//         Tensor shifted = logits - m;
//         Tensor exp_shifted = shifted.exp();
//         double log_sum_exp = std::log(exp_shifted.sum().item());
//
//         double loss = -(one_hot * logits).sum().item() + m + log_sum_exp;
//         Tensor softmax = exp_shifted / exp_shifted.sum();
//         Tensor grad = softmax - one_hot;
//         return {loss, grad};
//     }
// };

class BinaryCrossEntropyLoss {
  public:
    std::pair<double, Tensor> operator()(const Tensor &probs, const Tensor &labels) const {
        Tensor clipped_probs =
            probs.map([](double x) { return std::min(std::max(x, 1e-7), 1 - 1e-7); }).squeeze();
        Tensor flattened_labels = labels.squeeze();

        double loss = -(flattened_labels.dot(clipped_probs.log()).item() +
                        (1 - flattened_labels).dot((1 - clipped_probs).log()).item());

        Tensor grads = probs - labels;
        return {loss, grads};
    }
};

#endif
