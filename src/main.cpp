#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "nn.h"
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


class DataLoader {
public:
    std::mt19937 gen;
    std::bernoulli_distribution d;

    DataLoader() {
        std::random_device rd;
        this->gen = std::mt19937{ rd() };
        this->d = std::bernoulli_distribution{0.5};
    }

    std::pair<Tensor, Tensor> next() {
      bool a = d(gen);
      bool b = d(gen);

      std::vector<double> out(2);
      out[(float) (a^b)] = 1;

      return {
        Tensor::from_vec({(float) a, (float) b}),
        Tensor::from_vec(out)
      };
    }
};


double is_correct(const Tensor& output, const Tensor& label) {
  double max = output.at({0});
  int max_idx = 0;

  for (int i = 1; i < output.shape[0]; ++i) {
    if (output.at({i}) > max) {
      max = output.at({i});
      max_idx = i;
    }
  }

  return label.at({max_idx}) == 1.0 ? 1.0 : 0.0;
}


int main() {
  int num_epochs = 5;
  double learning_rate = 0.1;
  int batch_size = 16;

  DataLoader dl{};
  MLP model(2, 8, 2);
  CrossEntropyLoss criterion{};

  double avg_accuracy = 0.0;
  double avg_loss = 0.0;

  int iter = 0;
  int epoch = 0;

  while (true) {
    auto [x, y] = dl.next();

    Tensor logits = model.forward(x);
    auto [loss, loss_grads] = criterion(logits, y);

    model.backward(loss_grads);

    avg_accuracy += is_correct(logits, y);
    avg_loss += loss;

    if (iter % batch_size == 0) {
      epoch += 1;

      model.update(learning_rate);
      model.zero_grad();

      std::cout << epoch 
            << "\tLoss: " << avg_loss / batch_size
            << "\tAccuracy: " << avg_accuracy / batch_size
            << std::endl;

      if (epoch == num_epochs) {
        break;
      }

      avg_accuracy = 0.0;
      avg_loss = 0.0;
    }

    iter += 1;
  }
}
