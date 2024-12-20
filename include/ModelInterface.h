#ifndef MODELINTERFACE_H
#define MODELINTERFACE_H

#include <vector>

#include "Layer.h"
#include "Matrix.h"

// Model Class to Manage Layers
class Model {
 private:
  std::vector<Layer*> layers;

 public:
  void addLayer(Layer* layer_ptr) { layers.push_back(std::move(layer_ptr)); }

  // Convenience method for creating layers inline
  template <typename LayerType, typename... Args>
  void addLayer(Args&&... args) {
    layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
  }

  Matrix predict(const Matrix& input) {
    std::vector<Matrix> matrixVec(1, input);
    for (auto& layer : layers) {
      matrixVec = layer->forward(matrixVec);
    }
    return matrixVec[0];
  }

  void train(const Matrix& input, const Matrix& target) {
    size_t layer_num = layers.size();

    // Forward pass
    std::vector<Matrix> matrixVec(1, input);
    for (size_t i = 0; i < layer_num; i++) {
      matrixVec = layers[i]->forward(matrixVec);
    }

    // Backward pass (backpropagation)
    std::vector<Matrix> gradientVec(1, computeGradient(matrixVec[0], target));
    for (size_t i = layer_num - 1; i >= 0; i--) {
      gradientVec = layers[i]->backward(gradientVec);
    }
  }

 private:
  Matrix computeGradient(const Matrix& prediction, const Matrix& target) {
    // Cross Entropy
    return prediction - target;
  }
};

#endif