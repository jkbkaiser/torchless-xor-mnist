#include <cmath>
#include <utility>
#include <optional>

#include "tensor.h"


// Base class for all neural network modules
class Module {
public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor grads) = 0;
    virtual void update(double learning_rate) = 0;
    virtual void zero_grad() = 0;
    virtual ~Module() = default;
};


// Implemets a fully connected linear layer
class Linear : public Module {
public:
    Tensor weight;
    Tensor bias;

    std::optional<Tensor> cached_input;
    Tensor weight_grads;
    Tensor bias_grads;

    Linear(int in_features, int out_features)
        : weight(Tensor::rand({in_features, out_features})),
          bias(Tensor::zeros({out_features})),
          weight_grads(Tensor::zeros({in_features, out_features})),
          bias_grads(Tensor::zeros({out_features}))
    {}

    Tensor forward(Tensor input) override {
        this->cached_input = input;
        return input.matmul(weight) + bias;
    }

    Tensor backward(Tensor grad_output) override {
        if (!cached_input) {
            throw std::runtime_error("You must first call the modules before calling `backward`");
        }

        // Bias gradients
        bias_grads = bias_grads + grad_output;

        // Weight gradients
        Tensor go = grad_output.add_dim(0);
        Tensor ci = cached_input.value().add_dim(1);

        weight_grads = weight_grads + ci.matmul(go);

        // Input gradients
        Tensor input_grads = grad_output.matmul(weight.transpose());

        return input_grads;
    }

    void update(double learning_rate) override {
        std::cout << "updating weights" << std::endl;

        std::cout << "grads " << weight_grads << std::endl;

        std::cout << "before" << weight << std::endl;

        weight = weight - (learning_rate * weight_grads);

        std::cout << "after" << weight << std::endl;

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
    Tensor forward(Tensor input) override {
        return input.map([](double x) { return x > 0 ? x : 0; } );
    }

    Tensor backward(Tensor grads) override {
        return grads.map([](double x) { return x > 0 ? 1 : 0; } );
    }

    void update(double learning_rate) override {}
    void zero_grad() override {}
};


// Combines softmax computation with cross entropy loss
class CrossEntropyLoss {
public:
    std::pair<double, Tensor> operator()(const Tensor& pred, const Tensor& label) const {
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


// Simple MLP with one hidden layer and the ReLU activation function
class MLP : public Module {
public:
    Linear l1;
    Linear l2;
    ReLU relu{};

    MLP(int in_features, int hidden_features, int out_features)
        : l1({in_features, hidden_features}),
          l2({hidden_features, out_features})
    {}

    Tensor forward(Tensor input) override {
        Tensor l1_output = l1.forward(input);
        Tensor hidden = relu.forward(l1_output);
        Tensor l2_output = l2.forward(hidden);
        return l2_output;
    }

    Tensor backward(Tensor grad_output) override {
        Tensor l2_grad = l2.backward(grad_output);
        Tensor hidden = relu.backward(l2_grad);
        Tensor l1_grad = l1.backward(hidden);
        return l1_grad;
    }

    void update(double learning_rate) override {
        l1.update(learning_rate);
        l2.update(learning_rate);
    }


    void zero_grad() override {
        l1.zero_grad();
        l2.zero_grad();
    }
};
