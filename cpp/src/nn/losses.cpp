#include <cmath>

#include <torchless/nn.h>
#include <torchless/tensor.h>

std::pair<double, Tensor> CrossEntropyLoss::operator()(const Tensor &logits,
                                                       const Tensor &labels) const {
    size_t batch_size = logits.shape[0];
    size_t num_classes = logits.shape[1];
    Tensor flattened_labels = labels.squeeze();

    double total_loss = 0.0;
    Tensor grad = Tensor::zeros({batch_size, num_classes});

    // Compute loss and gradients per sample
    for (size_t i = 0; i < batch_size; ++i) {
        size_t true_class = static_cast<size_t>(flattened_labels.at({i}));

        // Find max logit for numerical stability
        double max_logit = logits.at({i, 0});
        for (size_t c = 1; c < num_classes; ++c) {
            max_logit = std::max(max_logit, logits.at({i, c}));
        }

        // Compute log_sum_exp for this sample
        double sum_exp = 0.0;
        for (size_t c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(logits.at({i, c}) - max_logit);
        }
        double log_sum_exp = std::log(sum_exp);

        // Loss for this sample: -log(p_true_class)
        double sample_loss = -(logits.at({i, true_class}) - max_logit - log_sum_exp);
        total_loss += sample_loss;

        // Gradient: softmax - one_hot
        for (size_t c = 0; c < num_classes; ++c) {
            double softmax_c = std::exp(logits.at({i, c}) - max_logit - log_sum_exp);
            double one_hot_c = (c == true_class) ? 1.0 : 0.0;
            grad.at({i, c}) = (softmax_c - one_hot_c) / batch_size;
        }
    }

    return {total_loss / batch_size, grad};
}

std::pair<double, Tensor> BinaryCrossEntropyLoss::operator()(const Tensor &logits,
                                                             const Tensor &labels) const {
    Tensor clipped_probs =
        logits.map([](double x) { return std::min(std::max(x, 1e-7), 1 - 1e-7); }).squeeze();
    Tensor flattened_labels = labels.squeeze();

    double loss = -(flattened_labels.dot(clipped_probs.log()).item() +
                    (1 - flattened_labels).dot((1 - clipped_probs).log()).item());

    Tensor grads = logits - labels;
    return {loss, grads};
}
