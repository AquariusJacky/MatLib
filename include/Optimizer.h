#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

#include "Matrix.h"

class Adam {
 private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  std::vector<std::vector<float>> m;  // First moment
  std::vector<std::vector<float>> v;  // Second moment
  int t;                              // Time step

 public:
  Adam(float lr = 0.001, float b1 = 0.9, float b2 = 0.999, float eps = 1e-8)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

  void initialize(const std::vector<std::vector<float>>& weights) {
    m = std::vector<std::vector<float>>(
        weights.size(), std::vector<float>(weights[0].size(), 0.0));
    v = std::vector<std::vector<float>>(
        weights.size(), std::vector<float>(weights[0].size(), 0.0));
  }

  void update(std::vector<std::vector<float>>& weights,
              const std::vector<std::vector<float>>& gradients) {
    t++;

    // If not initialized, initialize moment vectors
    if (m.empty()) {
      initialize(weights);
    }

    for (size_t i = 0; i < weights.size(); i++) {
      for (size_t j = 0; j < weights[i].size(); j++) {
        // Update biased first moment estimate
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradients[i][j];

        // Update biased second raw moment estimate
        v[i][j] =
            beta2 * v[i][j] + (1 - beta2) * gradients[i][j] * gradients[i][j];

        // Compute bias-corrected first moment estimate
        float m_hat = m[i][j] / (1 - std::pow(beta1, t));

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[i][j] / (1 - std::pow(beta2, t));

        // Update parameters
        weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    }
  }
};

#endif