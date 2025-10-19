#include <cmath>
#include <utility>

#include "tensor.h"

class CrossEntropyLoss {
  public:
    std::pair<double, Tensor> operator()(const Tensor &pred, const Tensor &label) const {
        double m = pred.max().item();
        Tensor shifted = pred - m;
        Tensor exp_shifted = shifted.exp();
        double log_sum_exp = std::log(exp_shifted.sum().item());

        double loss = -(label.dot(pred)).item() + m + log_sum_exp;
        Tensor softmax = exp_shifted / exp_shifted.sum();
        Tensor grad = softmax - label;
        return {loss, grad};
    }
};

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
