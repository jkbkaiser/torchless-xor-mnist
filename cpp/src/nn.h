#ifndef NN_H
#define NN_H

#include <cmath>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>

#include "tensor.h"

template <typename T> std::string to_string(const std::vector<T> &v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// Base class for all neural network modules
class Module {
  public:
    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor grads) = 0;
    virtual void update(double learning_rate) = 0;
    virtual void zero_grad() = 0;
    virtual ~Module() = default;
};

// Implements a fully connected linear layer
class Linear : public Module {
  public:
    Tensor weight;
    Tensor bias;

    std::optional<Tensor> cached_input;
    Tensor weight_grads;
    Tensor bias_grads;

    Linear(size_t in_features, size_t out_features)
        : weight(Tensor::rand({out_features, in_features}) * 2.0 *
                     std::sqrt(6.0 / (in_features + out_features)) -
                 std::sqrt(6.0 / (in_features + out_features))),
          bias(Tensor::zeros({out_features})),
          weight_grads(Tensor::zeros({out_features, in_features})),
          bias_grads(Tensor::zeros({out_features})) {}

    Tensor forward(Tensor x) override {
        this->cached_input = x;
        return x.matmul(weight.transpose()) + bias;
    }

    Tensor backward(Tensor grad_output) override {
        if (!this->cached_input) {
            throw std::runtime_error("You must first call the modules before calling `backward`");
        }
        size_t b = grad_output.shape[0];
        this->bias_grads = grad_output.sum(1) / b;
        this->weight_grads = grad_output.transpose().matmul(this->cached_input.value()) / b;
        Tensor input_grads = grad_output.matmul(this->weight);
        return input_grads;
    }

    void update(double learning_rate) override {
        weight = weight - (learning_rate * weight_grads);
        bias = bias - (learning_rate * bias_grads);
    }

    void zero_grad() override {
        weight_grads = Tensor::zeros(weight_grads.shape);
        bias_grads = Tensor::zeros(bias_grads.shape);
    }
};

// ReLU activation function
class ReLU : public Module {
  public:
    std::optional<Tensor> cached;

    Tensor forward(Tensor x) override {
        cached = x.map([](double x) { return x > 0 ? x : 0; });
        return cached.value();
    }

    Tensor backward(Tensor grads) override {
        if (!cached) {
            throw std::runtime_error("You must first call forward before calling backward");
        }
        // Element-wise multiplication: grads * (cached > 0)
        Tensor result(grads.shape);
        for (size_t i = 0; i < grads.data.size(); ++i) {
            result.data[i] = grads.data[i] * (cached.value().data[i] > 0 ? 1.0 : 0.0);
        }
        return result;
    }

    void update(double) override {}
    void zero_grad() override {}
};

// Simple MLP with one hidden layer and the ReLU activation function
class MLP : public Module {
  public:
    std::vector<std::unique_ptr<Module>> layers{};

    MLP(int in_features, int hidden_features, int out_features) {
        layers.push_back(std::make_unique<Linear>(in_features, hidden_features));
        layers.push_back(std::make_unique<ReLU>());
        layers.push_back(std::make_unique<Linear>(hidden_features, out_features));
    }

    Tensor forward(Tensor x) override {
        for (auto &layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    Tensor backward(Tensor grad_output) override {
        for (int i = layers.size() - 1; i >= 0; --i) {
            grad_output = layers[i]->backward(grad_output);
        }
        return grad_output;
    }

    void update(double learning_rate) override {
        for (auto &layer : layers) {
            layer->update(learning_rate);
        }
    }

    void zero_grad() override {
        for (auto &layer : layers) {
            layer->zero_grad();
        }
    }
};

#endif
