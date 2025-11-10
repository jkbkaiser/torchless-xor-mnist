#include <cmath>

#include <torchless/nn.h>
#include <torchless/tensor.h>

template <Device D>
Linear<D>::Linear(size_t in_features, size_t out_features, std::mt19937 rng)
    : weight_(Tensor<D>::rand({out_features, in_features}, rng) * 2.0 *
                  std::sqrt(6.0 / (in_features + out_features)) -
              std::sqrt(6.0 / (in_features + out_features))),
      bias_(Tensor<D>::zeros({out_features})),
      weight_grads_(Tensor<D>::zeros({out_features, in_features})),
      bias_grads_(Tensor<D>::zeros({out_features})) {}

template <Device D> Tensor<D> Linear<D>::forward(Tensor<D> x) {
    cached_input_ = x;
    return x.matmul(weight_.transpose()) + bias_;
}

template <Device D> Tensor<D> Linear<D>::backward(Tensor<D> grad_output) {
    if (!cached_input_) {
        throw std::runtime_error("You must first call the modules before calling `backward`");
    }

    size_t b = grad_output.shape[0];
    bias_grads_ = grad_output.sum(1) / b;
    weight_grads_ = grad_output.transpose().matmul(cached_input_.value()) / b;
    Tensor input_grads = grad_output.matmul(weight_);
    return input_grads;
}

template <Device D> void Linear<D>::update(double learning_rate) {
    weight_ = weight_ - (learning_rate * weight_grads_);
    bias_ = bias_ - (learning_rate * bias_grads_);
}

template <Device D> void Linear<D>::zero_grad() {
    weight_grads_ = Tensor<D>::zeros(weight_grads_.shape);
    bias_grads_ = Tensor<D>::zeros(bias_grads_.shape);
}

template <Device D> Tensor<D> ReLU<D>::forward(Tensor<D> x) {
    cached_ = x.map([](double x) { return x > 0 ? x : 0; });
    return cached_.value();
}

template <Device D> Tensor<D> ReLU<D>::backward(Tensor<D> grads) {
    if (!cached_) {
        throw std::runtime_error("You must first call forward before calling backward");
    }

    Tensor result(grads.shape);
    for (size_t i = 0; i < grads.data.size(); ++i) {
        result.data[i] = grads.data[i] * (cached_.value().data[i] > 0 ? 1.0 : 0.0);
    }
    return result;
}

template <Device D> void ReLU<D>::update(double) {}

template <Device D> void ReLU<D>::zero_grad() {}

template <Device D>
MLP<D>::MLP(int in_features, int hidden_features, int out_features, std::mt19937 rng) {
    layers_.push_back(std::make_unique<Linear<D>>(in_features, hidden_features, rng));
    layers_.push_back(std::make_unique<ReLU<D>>());
    layers_.push_back(std::make_unique<Linear<D>>(hidden_features, out_features, rng));
}

template <Device D> Tensor<D> MLP<D>::forward(Tensor<D> x) {
    for (auto &layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

template <Device D> Tensor<D> MLP<D>::backward(Tensor<D> grad_output) {
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad_output = layers_[i]->backward(grad_output);
    }
    return grad_output;
}

template <Device D> void MLP<D>::update(double learning_rate) {
    for (auto &layer : layers_) {
        layer->update(learning_rate);
    }
}

template <Device D> void MLP<D>::zero_grad() {
    for (auto &layer : layers_) {
        layer->zero_grad();
    }
}
