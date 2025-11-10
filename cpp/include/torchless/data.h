#ifndef TORCHLESS_DATA_H
#define TORCHLESS_DATA_H

#include <filesystem>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <torchless/tensor.h>

enum Split { TRAIN, TEST };

class Dataset {
  public:
    virtual ~Dataset() = default;
    virtual std::pair<std::vector<double>, double> sample(size_t idx) const = 0;
    virtual size_t size() const = 0;
};

class XORDataset : public Dataset {
  public:
    XORDataset(size_t num_samples, double noise_std, std::mt19937 rng);

    std::pair<std::vector<double>, double> sample(size_t idx) const override;
    size_t size() const override;

  private:
    size_t num_samples_;
    double noise_std_;

    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> dist_;
};

class MNISTDataset : public Dataset {
  public:
    MNISTDataset(Split split, const std::string &dir);

    std::pair<std::vector<double>, double> sample(size_t idx) const override;
    size_t size() const override;

  private:
    Split split_;
    std::filesystem::path dir_;
    size_t num_samples_;
    std::vector<std::vector<double>> images_;
    std::vector<double> labels_;
};

class Dataloader {
  public:
    Dataloader(const Dataset *ds, size_t batch_size, std::mt19937 rng);

    class Iterator {
      private:
        const Dataset *ds_;
        const std::vector<size_t> *indices_;
        size_t batch_size_;
        size_t pos_;

      public:
        using value_type = std::pair<Tensor<Device::CPU>, Tensor<Device::CPU>>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        Iterator(const Dataset *ds, const std::vector<size_t> *indices, size_t batch_size,
                 size_t pos);

        value_type operator*() const;
        Iterator &operator++();
        bool operator!=(const Iterator &other) const;
    };

    Iterator begin() const;
    Iterator end() const;

  private:
    const Dataset *ds_;
    size_t batch_size_;

    mutable std::vector<size_t> indices_;
    mutable std::mt19937 rng_;
};

#endif
