#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

#include "tensor.h"

enum Split { TRAIN, TEST };

class Dataset {
  public:
    virtual ~Dataset() = default;
    virtual std::pair<Tensor, Tensor> sample(size_t idx) const = 0;
    virtual size_t size() const = 0;
};

class XORDataSet : public Dataset {
  private:
    size_t num_samples_;
    double noise_std_;

  public:
    XORDataSet(size_t num_samples, double noise_std)
        : num_samples_(num_samples), noise_std_(noise_std) {}

    std::pair<Tensor, Tensor> sample(size_t idx) const {
        // This can be improved
        Tensor a = Tensor::randint(0, 2, {1});
        Tensor b = Tensor::randint(0, 2, {1});

        Tensor y = a ^ b;
        Tensor x = stack({a, b}, 1).squeeze();
        Tensor noise = Tensor::rand(x.shape) * this->noise_std_;

        return {x + noise, y};
    }

    size_t size() const { return this->num_samples_; }
};

class Dataloader {
  private:
    const Dataset *ds_;
    size_t batch_size_;
    mutable std::vector<size_t> indices_;
    mutable std::mt19937 gen_;

  public:
    Dataloader(const Dataset *ds, size_t batch_size, std::mt19937::result_type seed = 0)
        : ds_(ds), batch_size_(batch_size), gen_(seed) {
        indices_.resize(ds->size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    class Iterator {
      private:
        const Dataset *ds_;
        const std::vector<size_t> *indices_;
        size_t batch_size_;
        size_t pos_;

      public:
        using value_type = std::pair<Tensor, Tensor>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        Iterator(const Dataset *ds, const std::vector<size_t> *indices, size_t batch_size,
                 size_t pos)
            : ds_(ds), indices_(indices), batch_size_(batch_size), pos_(pos) {}

        value_type operator*() const {
            std::vector<Tensor> xs{};
            std::vector<Tensor> ys{};
            size_t end = std::min(pos_ + batch_size_, indices_->size());

            for (size_t i = pos_; i < end; ++i) {
                size_t idx = (*indices_)[i];
                auto [x, y] = ds_->sample(idx);
                xs.push_back(x);
                ys.push_back(y);
            }

            return {stack(xs, 0), stack(ys, 0)};
        }

        Iterator &operator++() {
            pos_ = std::min(pos_ + batch_size_, indices_->size());
            return *this;
        }

        bool operator!=(const Iterator &other) const { return pos_ != other.pos_; }
    };

    Iterator begin() const {
        std::shuffle(indices_.begin(), indices_.end(), this->gen_);
        return Iterator(ds_, &indices_, batch_size_, 0);
    }

    Iterator end() const { return Iterator(ds_, nullptr, batch_size_, indices_.size()); }
};

// Reshuffle every begin() call
// class MNISTDataLoader {
// public:
//     std::mt19937 gen;
//     int batch_size;
//     Split split;
//
//     MNISTDataLoader(int batch_size, Split split) {
//         std::random_device rd;
//         this->gen = std::mt19937{ rd() };
//         this->batch_size = batch_size;
//         this->split = split;
//     }
//
//     std::pair<Tensor, Tensor> next() {
//       Tensor a = Tensor::randint(0, 2, {this->batch_size});
//       Tensor b = Tensor::randint(0, 2, {this->batch_size});
//
//       Tensor y = a^b;
//       Tensor x = stack({a, b}, 1);
//       Tensor noise = Tensor::rand(x.shape) * this->noise_std;
//
//       return {x + noise, y};
//     }
// };
