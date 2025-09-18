#include<cmath>
#include<iostream>
#include<iomanip>
#include<vector>

#include "tensor.h"
#include "nn.h"

#define ASSERT_TRUE(cond) \
  if (!(cond)) { \
    std::cout << "\n\tAssertion failed: " << #cond << std::endl; \
    return 1; \
  }

#define ASSERT_FALSE(cond) \
  if (cond) { \
    std::cout << "\n\tAssertion failed: " << #cond << std::endl; \
    return 1; \
  }

int test_eye() {
  Tensor a = Tensor::eye({2, 2});
  ASSERT_TRUE(a.at({0, 0}) == 1.0);
  ASSERT_TRUE(a.at({0, 1}) == 0.0);
  ASSERT_TRUE(a.at({1, 0}) == 0.0);
  ASSERT_TRUE(a.at({1, 1}) == 1.0);
  return 0;
}

int test_transpose() {
  Tensor a = Tensor({2, 2});
  a.at({0, 0}) = 1.0;
  a.at({0, 1}) = 2.0;
  a.at({1, 0}) = 3.0;
  a.at({1, 1}) = 4.0;

  Tensor b = a.transpose();
  ASSERT_TRUE(b.at({0, 0}) == 1.0);
  ASSERT_TRUE(b.at({0, 1}) == 3.0);
  ASSERT_TRUE(b.at({1, 0}) == 2.0);
  ASSERT_TRUE(b.at({1, 1}) == 4.0);
  return 0;
}

int test_equal() {
  Tensor a = Tensor::zeros({1});
  Tensor b = Tensor::zeros({1});
  ASSERT_TRUE(a == b);
  return 0;
}

int test_scalar_add() {
  Tensor a = Tensor::zeros({1, 2});
  Tensor b = Tensor::ones({1, 2});
  ASSERT_TRUE(a + 1 == b);
  ASSERT_FALSE(a + 2 == b);
  ASSERT_TRUE(1 + a == b);
  return 0;
}

int test_scalar_subtract() {
  Tensor a = Tensor::ones({1, 2});
  Tensor b = Tensor::zeros({1, 2});
  ASSERT_TRUE(a - 1 == b);
  ASSERT_FALSE(a - 2 == b);
  ASSERT_TRUE(1 - a == b);
  return 0;
}

int test_scalar_multiply() {
  Tensor a = Tensor::ones({1, 2});
  Tensor b = Tensor::filled({1, 2}, 2.0);
  ASSERT_TRUE(a * 2 == b);
  ASSERT_FALSE(a * 3 == b);
  ASSERT_TRUE(2 * a == b);
  return 0;
}


int test_scalar_divide() {
  Tensor a = Tensor::filled({1, 2}, 6.0);
  Tensor b = Tensor::filled({1, 2}, 2.0);
  Tensor c = Tensor::filled({1, 2}, 3.0);
  ASSERT_TRUE(a / 3 == b);
  ASSERT_FALSE(a / 2 == b);
  ASSERT_TRUE(6 / b == c);
  return 0;
}

int test_broadcast_add() {
  Tensor a = Tensor::filled({4}, 6.0);
  Tensor b = Tensor::filled({4}, 2.0);
  Tensor c = stack({a, b});

  Tensor d = Tensor::filled({1}, 1.0);

  Tensor e = c + d;

  ASSERT_TRUE(e.at({0, 0}) == 7.0);
  ASSERT_TRUE(e.at({1, 0}) == 3.0);

  Tensor f = Tensor::filled({4}, 1.0);
  f.at({1}) = 2.0;
  f.at({2}) = 3.0;
  f.at({3}) = 4.0;

  Tensor g = c + f;

  Tensor h = Tensor::from_vec(
    std::vector<std::vector<double>>{
      {7.0, 8.0, 9.0, 10.0},
      {3.0, 4.0, 5.0, 6.0},
    }
  );

  ASSERT_TRUE(g == h)


  return 0;
}

int test_dot() {
  Tensor a = Tensor::filled({2}, 2.0);
  Tensor b = Tensor::filled({2}, 3.0);
  ASSERT_TRUE(a.dot(b).item() == 12.0);
  return 0;
}

int test_matmul() {
  // Dot product
  Tensor a = Tensor::filled({2}, 2.0);
  Tensor b = Tensor::filled({2}, 3.0);
  ASSERT_TRUE(a.matmul(b).item() == 12.0);

  // Matrix Multiplication
  Tensor c = Tensor::filled({2, 2}, 2.0);
  Tensor d = Tensor::filled({2, 2}, 3.0);
  Tensor e = Tensor::filled({2, 2}, 12.0);
  ASSERT_TRUE(c.matmul(d) == e);

  return 0;
}

int test_stack() {
  Tensor a = Tensor::filled({2}, 1.0);
  Tensor b = Tensor::filled({2}, 2.0);
  Tensor c = Tensor::filled({2}, 3.0);
  Tensor d = Tensor::filled({2}, 4.0);

  std::vector<Tensor> tensors = {a, b, c, d};
  Tensor e = stack(tensors);

  ASSERT_TRUE(e.at({0, 0}) == 1.0);
  ASSERT_TRUE(e.at({1, 0}) == 2.0);
  ASSERT_TRUE(e.at({2, 0}) == 3.0);
  ASSERT_TRUE(e.at({3, 0}) == 4.0);

  Tensor f = Tensor::ones({4});
  Tensor g = Tensor::filled({4}, 2.0);

  tensors = {f, g};
  Tensor h = stack(tensors);

  ASSERT_TRUE(h.at({0, 0}) == 1.0);
  ASSERT_TRUE(h.at({1, 0}) == 2.0);

  return 0;
}

int test_linear() {
  Linear l = Linear(2, 8);

  l.weight = Tensor::from_vec(
    std::vector<std::vector<double>>{
      {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
      {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0},
    }
  );
  l.bias = Tensor::ones({8});

  Tensor input = Tensor::from_vec({1, 2});

  Tensor output = l.forward(input);

  Tensor expected_output = Tensor::from_vec(
    std::vector<double>{
      {22.0, 43.0, 64.0, 85.0, 106.0, 127.0, 148.0, 169.0},
    }
  );

  ASSERT_TRUE(output == expected_output)

  return 0;
}

int test_relu() {
  Tensor input = Tensor::from_vec(
    std::vector<double>{
      -3.0, 4.0, -1000.0, 0.0001,
    }
  );

  ReLU relu{};

  Tensor output = relu.forward(input);

  Tensor expected_output = Tensor::from_vec(
    std::vector<double>{
      0, 4.0, 0, 0.0001,
    }
  );

  ASSERT_TRUE(output == expected_output)

  return 0;
}


int test_cross_entropy() {
    CrossEntropyLoss criterion;

    Tensor pred = Tensor::from_vec({2.0, 1.0});  // example logits
    Tensor label = Tensor::from_vec({1.0, 0.0});
    auto [loss, grad] = criterion(pred, label);
    double expected_loss = 0.3132617;

    ASSERT_TRUE(std::abs(loss - expected_loss) < 1e-5);

    Tensor softmax = (pred - pred.max()).exp();
    softmax = softmax / softmax.sum();

    Tensor expected_grad = softmax - label;

    for (int i = 0; i < grad.shape[0]; ++i) {
        ASSERT_TRUE(std::abs(grad.at({i}) - expected_grad.at({i})) < 1e-5);
    }

    return 0;
}

int main() {
  std::cout << "Running tests\n" << std::endl;

  std::vector<std::pair<std::string, std::function<int()>>> tests = {
    {"test_equal", test_equal},
    {"test_scalar_add", test_scalar_add},
    {"test_scalar_subtract", test_scalar_subtract},
    {"test_scalar_multiply", test_scalar_multiply},
    {"test_scalar_divide", test_scalar_divide},
    {"test_eye", test_eye},
    {"test_dot", test_dot},
    {"test_matmul", test_matmul},
    {"test_transpose", test_transpose},
    {"test_stack", test_stack},
    {"test_broadcast_add", test_broadcast_add},
    {"test_linear", test_linear},
    {"test_relu", test_relu},
    {"test_cross_entropy", test_cross_entropy},
  };

  int failures = 0;

  size_t max_name_length = 0;
  for (const auto& [name, _] : tests) {
    if (name.length() > max_name_length) max_name_length = name.length();
  }

  for (auto& [name, test] : tests) {
    std::cout << "Running " << std::left << std::setw(max_name_length + 2) << name << "...";
    int result = test();
    if (result != 0) {
      failures++;
    } else {
      std::cout << "\t\t✅" << std::endl;
    }
  }

  std::cout << "\n" << (failures == 0 ? "All tests passed." : std::to_string(failures) + " test(s) failed.") << std::endl;

  return 0;
}
