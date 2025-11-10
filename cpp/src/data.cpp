#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <torchless/data.h>
#include <torchless/tensor.h>

XORDataset::XORDataset(size_t num_samples, double noise_std, std::mt19937 rng)
    : num_samples_(num_samples), noise_std_(noise_std), rng_(rng), dist_(0.0, 1.0) {}

std::pair<std::vector<double>, double> XORDataset::sample(size_t /*idx*/) const {
    double a = dist_(rng_) < 0.5 ? 0.0 : 1.0;
    double b = dist_(rng_) < 0.5 ? 0.0 : 1.0;

    double y = static_cast<int>(a) ^ static_cast<int>(b);

    double noise_a = dist_(rng_) * noise_std_;
    double noise_b = dist_(rng_) * noise_std_;

    return {{a + noise_a, b + noise_b}, y};
}

size_t XORDataset::size() const { return num_samples_; }

uint32_t read_be_uint32(std::ifstream &fs) {
    uint8_t bytes[4];
    fs.read(reinterpret_cast<char *>(bytes), 4);

    // Convert big endian to little endian
    uint32_t result = (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
                      (uint32_t(bytes[2]) << 8) | (uint32_t(bytes[3]));

    return result;
}

std::vector<std::vector<double>> load_images(const std::filesystem::path &idx_file_path) {
    std::ifstream file(idx_file_path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open " + idx_file_path.string());
    }

    // Skip magic number
    read_be_uint32(file);
    uint32_t n_images = read_be_uint32(file);
    uint32_t n_rows = read_be_uint32(file);
    uint32_t n_cols = read_be_uint32(file);

    size_t image_size = n_rows * n_cols;
    std::vector<uint8_t> raw_images(n_images * image_size);
    file.read(reinterpret_cast<char *>(raw_images.data()), raw_images.size());

    std::vector<double> flat_images(n_images * image_size);
    std::transform(raw_images.begin(), raw_images.end(), flat_images.begin(),
                   [](uint8_t x) { return static_cast<double>(x); });

    std::vector<std::vector<double>> images(n_images, std::vector<double>(image_size));
    for (size_t i = 0; i < n_images; ++i) {
        std::copy(flat_images.begin() + i * image_size, flat_images.begin() + (i + 1) * image_size,
                  images[i].begin());
    }

    return images;
}

std::vector<double> load_labels(const std::filesystem::path &idx_file_path) {
    // The file layout for the IDX format is given in `idx_format.md`.
    std::ifstream file(idx_file_path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open " + idx_file_path.string());
    }

    // Skip magic number
    read_be_uint32(file);
    uint32_t n_labels = read_be_uint32(file);

    std::vector<uint8_t> raw_labels(n_labels);
    file.read(reinterpret_cast<char *>(raw_labels.data()), raw_labels.size());

    std::vector<double> labels(raw_labels.size());
    std::transform(raw_labels.begin(), raw_labels.end(), labels.begin(),
                   [](uint8_t x) { return static_cast<double>(x); });

    return labels;
}

// Load MNIST dataset from IDX file format
// IDX Format (big-endian):
//   Images: [magic][n_images][n_rows][n_cols][pixel_data...]
//     - magic: 4 bytes (0x00000803 for images)
//     - dimensions: 4 bytes each (uint32)
//     - pixels: n_images * n_rows * n_cols bytes (uint8, 0-255)
//   Labels: [magic][n_labels][label_data...]
//     - magic: 4 bytes (0x00000801 for labels)
//     - n_labels: 4 bytes (uint32)
//     - labels: n_labels bytes (uint8, 0-9)
MNISTDataset::MNISTDataset(Split split, const std::string &dir) : split_(split), dir_(dir) {
    if (split_ == TRAIN) {
        std::filesystem::path img_file_path = dir_ / "train-images.idx3-ubyte";
        std::filesystem::path label_file_path = dir_ / "train-labels.idx1-ubyte";

        images_ = load_images(img_file_path);
        labels_ = load_labels(label_file_path);

        num_samples_ = labels_.size();
    } else if (split_ == TEST) {
        std::filesystem::path img_file_path = dir_ / "t10k-images.idx3-ubyte";
        std::filesystem::path label_file_path = dir_ / "t10k-labels.idx1-ubyte";

        images_ = load_images(img_file_path);
        labels_ = load_labels(label_file_path);

        num_samples_ = labels_.size();
    } else {
        throw std::runtime_error("Split must be TRAIN or TEST.");
    }
}

std::pair<std::vector<double>, double> MNISTDataset::sample(size_t idx) const {
    return {images_[idx], labels_[idx]};
}

size_t MNISTDataset::size() const { return num_samples_; }

Dataloader::Dataloader(const Dataset *ds, size_t batch_size, std::mt19937 rng)
    : ds_(ds), batch_size_(batch_size), rng_(rng) {
    indices_.resize(ds->size());
    std::iota(indices_.begin(), indices_.end(), 0);
}

Dataloader::Iterator::Iterator(const Dataset *ds, const std::vector<size_t> *indices,
                               size_t batch_size, size_t pos)
    : ds_(ds), indices_(indices), batch_size_(batch_size), pos_(pos) {}

Dataloader::Iterator::value_type Dataloader::Iterator::operator*() const {
    std::vector<std::vector<double>> xs{};
    std::vector<double> ys{};
    size_t end = std::min(pos_ + batch_size_, indices_->size());

    for (size_t i = pos_; i < end; ++i) {
        size_t idx = (*indices_)[i];
        auto [x, y] = ds_->sample(idx);
        xs.push_back(x);
        ys.push_back(y);
    }

    return {Tensor<Device::CPU>(xs), Tensor<Device::CPU>(ys).add_dim(1)};
}

Dataloader::Iterator &Dataloader::Iterator::operator++() {
    pos_ = std::min(pos_ + batch_size_, indices_->size());
    return *this;
}

bool Dataloader::Iterator::operator!=(const Dataloader::Iterator &other) const {
    return pos_ != other.pos_;
}

Dataloader::Iterator Dataloader::begin() const {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
    return Iterator(ds_, &indices_, batch_size_, 0);
}

Dataloader::Iterator Dataloader::end() const {
    return Iterator(ds_, nullptr, batch_size_, indices_.size());
}
