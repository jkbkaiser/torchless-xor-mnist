#ifndef TORCHLESS_TENSOR_BASE_H
#define TORCHLESS_TENSOR_BASE_H

#include <torchless/shape.h>

class BaseTensor {
  public:
    Shape shape_;

    BaseTensor(const Shape &shape_);

    virtual ~BaseTensor() = default;
    virtual float get(const std::vector<size_t> &indices) = 0;
};

#endif
