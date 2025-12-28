#ifndef TORCHLESS_SHAPE_H
#define TORCHLESS_SHAPE_H

#include <vector>

struct Shape {
    std::vector<size_t> dims_;

    Shape(std::initializer_list<size_t> list) : dims_(list) {}
    Shape(const std::vector<size_t> &vec) : dims_(vec) {}

    auto begin() { return dims_.begin(); }
    auto end() { return dims_.end(); }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }

    size_t size() const { return dims_.size(); }
    size_t numel() const {
        size_t total = 1;
        for (auto d : dims_)
            total *= d;
        return total;
    }

    size_t &operator[](size_t i) { return dims_[i]; }
    const size_t &operator[](size_t i) const { return dims_[i]; }

    bool operator==(const Shape &other) const { return dims_ == other.dims_; }

    bool operator!=(const Shape &other) const { return !(*this == other); }
};

#endif
