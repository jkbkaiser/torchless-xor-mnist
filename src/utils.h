#ifndef UTILS_H
#define UTILS_H

#include <iomanip>
#include <iostream>

inline void log_epoch(int epoch, double avg_loss, double avg_acc) {
    std::cout << "\rEpoch " << std::setw(3) << epoch << " | loss: " << std::fixed
              << std::setprecision(4) << avg_loss << " | acc: " << std::fixed
              << std::setprecision(2) << avg_acc * 100 << "%     " << std::endl;
}

#endif
