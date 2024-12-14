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
    Matrix currentInput = input;
    for (auto& layer : layers) {
      currentInput = layer->forward(currentInput);
    }
    return currentInput;
  }

  void train(const Matrix& input, const Matrix& target) {
    // Forward pass
    std::vector<Matrix> layerOutputs;
    Matrix currentInput = input;
    for (auto& layer : layers) {
      currentInput = layer->forward(currentInput);
      layerOutputs.push_back(currentInput);
    }

    // Backward pass (backpropagation)
    Matrix gradient = computeInitialGradient(currentInput, target);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
      gradient = (*it)->backward(gradient);
    }
  }

 private:
  Matrix computeInitialGradient(const Matrix& prediction,
                                const Matrix& target) {
    // Compute initial gradient based on loss function
    // For example, mean squared error
    return prediction - target;
  }
};

#endif