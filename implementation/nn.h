#include <memory>
#include <cmath>
#include <utility>
#include <ranges>
#include <optional>

#include "tensor.h"

template <typename T>
std::string to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 < v.size()) oss << ", ";
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


// Implemets a fully connected linear layer
class Linear : public Module {
public:
    Tensor weight;
    Tensor bias;

    std::optional<Tensor> cached_input;
    Tensor weight_grads;
    Tensor bias_grads;

    Linear(int in_features, int out_features)
        : weight(Tensor::rand({out_features, in_features}) * 2.0 * std::sqrt(6.0 / (in_features + out_features)) - std::sqrt(6.0 / (in_features + out_features))),
          bias(Tensor::zeros({out_features})),
          weight_grads(Tensor::zeros({out_features, in_features})),
          bias_grads(Tensor::zeros({out_features}))
    {}

    Tensor forward(Tensor x) override {
        this->cached_input = x;
        return x.matmul(weight.transpose()) + bias;
    }

    Tensor backward(Tensor grad_output) override {
        if (!this->cached_input) {
            throw std::runtime_error("You must first call the modules before calling `backward`");
        }

        // std::cout << "backward linear" << std::endl;
        // std::cout << grad_output << std::endl;

        // std::cout << this->weight.shape << std::endl;
        // std::cout << this->bias.shape << std::endl;
        // std::cout << grad_output.shape << std::endl;

        // self.b_grads = grads.sum(axis=0) / grads.shape[0]
        // self.w_grads = grads.T @ self.cached_input / grads.shape[0]
        // return grads @ self.w

        int b = grad_output.shape[0];

        // std::cout << b << std::endl;
        //
        // std::cout << "1" << std::endl;

        this->bias_grads = grad_output.sum(1) / b;
        // std::cout << "2" << grad_output.transpose().matmul(this->cached_input.value()) << b << std::endl;
        this->weight_grads = grad_output.transpose().matmul(this->cached_input.value()) / b;
        // std::cout << "3" << std::endl;

        // std::cout << "input gradients" << std::endl;

        // Input gradients
        Tensor input_grads = grad_output.matmul(this->weight);
        // std::cout << "4" << std::endl;

        // std::cout << "end" << std::endl;

        return input_grads;
    }

    void update(double learning_rate) override {
        // std::cout << "update" << std::endl;
        // std::cout << weight << std::endl;
        // std::cout << weight_grads << std::endl;
        weight = weight - (learning_rate * weight_grads);
        // std::cout << weight << std::endl;
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

    void update(double learning_rate) override {}
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
        for (auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    Tensor backward(Tensor grad_output) override {
        for (int i = layers.size() - 1; i >= 0; --i) {
            // std::cout << "\nblayer " << i << std::endl;
            // std::cout << "---" << std::endl;
            grad_output = layers[i]->backward(grad_output);
            // std::cout << "f" << std::endl;
        }
        return grad_output;
    }

    void update(double learning_rate) override {
        for (auto& layer : layers) {
            layer->update(learning_rate);
        }
    }

    void zero_grad() override {
        for (auto& layer : layers) {
            layer->zero_grad();
        }
    }
};
