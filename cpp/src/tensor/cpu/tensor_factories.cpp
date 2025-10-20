// Tensor factory functions - create new tensors
//
// Creation from scratch:
//   - zeros(shape)         - All zeros
//   - ones(shape)          - All ones
//   - filled(shape, value) - All same value
//   - rand(shape)          - Random values [0, 1)
//
// Creation from data:
//   - from_vec(vector<double>)
//   - from_vec(vector<double>, shape)
//   - from_vec(vector<vector<double>>)

#include <torchless/tensor.h>
