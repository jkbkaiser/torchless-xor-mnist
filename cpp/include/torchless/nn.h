#ifndef TORCHLESS_NN_H
#define TORCHLESS_NN_H

#include <memory>
#include <optional>
#include <random>

#include <torchless/tensor.h>

template <Device D> class Module {
  public:
    virtual ~Module() = default;

    virtual Tensor<D> forward(Tensor<D> x) = 0;
    virtual Tensor<D> backward(Tensor<D> grads) = 0;
    virtual void update(double learning_rate) = 0;
    virtual void zero_grad() = 0;
};

template <Device D> class Linear : public Module<D> {
  public:
    Tensor<D> weight_;
    Tensor<D> bias_;
    Tensor<D> weight_grads_;
    Tensor<D> bias_grads_;

    Linear(size_t in_features, size_t out_features, std::mt19937 rng);

    Tensor<D> forward(Tensor<D> x) override;
    Tensor<D> backward(Tensor<D> grad_output) override;
    void update(double learning_rate) override;
    void zero_grad() override;

  private:
    std::optional<Tensor<D>> cached_input_;
};

template <Device D> class ReLU : public Module<D> {
  public:
    Tensor<D> forward(Tensor<D> x) override;
    Tensor<D> backward(Tensor<D> grads) override;
    void update(double) override;
    void zero_grad() override;

  private:
    std::optional<Tensor<D>> cached_;
};

template <Device D> class MLP : public Module<D> {
  public:
    std::vector<std::unique_ptr<Module<D>>> layers_{};

    MLP(int in_features, int hidden_features, int out_features, std::mt19937 rng);

    Tensor<D> forward(Tensor<D> x) override;
    Tensor<D> backward(Tensor<D> grad_output) override;
    void update(double learning_rate) override;
    void zero_grad() override;
};

template <Device D> class CrossEntropyLoss {
  public:
    std::pair<double, Tensor<D>> operator()(const Tensor<D> &logits, const Tensor<D> &labels) const;
};

template <Device D> class BinaryCrossEntropyLoss {
  public:
    std::pair<double, Tensor<D>> operator()(const Tensor<D> &logits, const Tensor<D> &labels) const;
};

#endif
