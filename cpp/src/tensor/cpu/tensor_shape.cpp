// Shape manipulation operations
//
// Shape operations:
//   - squeeze()       - Remove dimensions of size 1
//   - add_dim(dim)    - Add dimension at position
//   - stack(tensors)  - Stack tensors along new axis
//
// Helper functions:
//   - (broadcasting helpers are in tensor_ops.cpp)

#include <torchless/tensor.h>
