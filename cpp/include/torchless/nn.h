#ifndef TORCHLESS_NN_H
#define TORCHLESS_NN_H

#include <memory>
#include <optional>

#include <torchless/tensor.h>

class Module {
  public:
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor grads) = 0;
    virtual void update(double learning_rate) = 0;
    virtual void zero_grad() = 0;
};

class Linear : public Module {
  public:
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grads_;
    Tensor bias_grads_;

    Linear(size_t in_features, size_t out_features);

    Tensor forward(Tensor x) override;
    Tensor backward(Tensor grad_output) override;
    void update(double learning_rate) override;
    void zero_grad() override;

  private:
    std::optional<Tensor> cached_input_;
};

class ReLU : public Module {
  public:
    Tensor forward(Tensor x) override;
    Tensor backward(Tensor grads) override;
    void update(double) override;
    void zero_grad() override;

  private:
    std::optional<Tensor> cached_;
};

class MLP : public Module {
  public:
    std::vector<std::unique_ptr<Module>> layers_{};

    MLP(int in_features, int hidden_features, int out_features);

    Tensor forward(Tensor x) override;
    Tensor backward(Tensor grad_output) override;
    void update(double learning_rate) override;
    void zero_grad() override;
};

class CrossEntropyLoss {
  public:
    std::pair<double, Tensor> operator()(const Tensor &logits, const Tensor &labels) const;
};

class BinaryCrossEntropyLoss {
  public:
    std::pair<double, Tensor> operator()(const Tensor &logits, const Tensor &labels) const;
};

#endif
