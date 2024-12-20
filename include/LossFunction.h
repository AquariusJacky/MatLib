#ifndef LOSSFUNTION_H
#define LOSSFUNTION_H

#include <cmath>
#include <vector>

#include "Matrix.h"

class CrossEntropyLoss {
 private:
  double epsilon = 1e-15;  // Small constant to avoid log(0)

 public:
  // Forward pass: compute loss
  double compute_loss(const std::vector<std::vector<double>>& predictions,
                      const std::vector<std::vector<double>>& targets) {
    double loss = 0.0;
    int batch_size = predictions.size();

    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < predictions[i].size(); j++) {
        // Clip predictions to avoid numerical instability
        double pred =
            std::max(std::min(predictions[i][j], 1 - epsilon), epsilon);
        loss -= targets[i][j] * std::log(pred);
      }
    }

    return loss / batch_size;
  }

  // Backward pass: compute gradients
  std::vector<std::vector<double>> compute_gradient(
      const std::vector<std::vector<double>>& predictions,
      const std::vector<std::vector<double>>& targets) {
    int batch_size = predictions.size();
    std::vector<std::vector<double>> gradients(
        predictions.size(), std::vector<double>(predictions[0].size(), 0.0));

    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < predictions[i].size(); j++) {
        double pred =
            std::max(std::min(predictions[i][j], 1 - epsilon), epsilon);
        gradients[i][j] = -targets[i][j] / pred / batch_size;
      }
    }

    return gradients;
  }
};

#endif