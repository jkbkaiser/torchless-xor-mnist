#ifndef TORCHLESS_UTILS_H
#define TORCHLESS_UTILS_H

#include <iomanip>
#include <iostream>

inline void log_epoch(int epoch, double avg_loss, double avg_acc) {
    std::cout << "\rEpoch " << std::setw(3) << epoch << " | loss: " << std::fixed
              << std::setprecision(4) << avg_loss << " | acc: " << std::fixed
              << std::setprecision(2) << avg_acc * 100 << "%     " << std::endl;
}

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << "(";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << ")";
    return os;
}

#endif
