#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "Matrix.h"

// Abstract base class for all layer types
class Layer {
 public:
  virtual ~Layer() = default;

  // Pure virtual method to be implemented by derived layers
  virtual std::vector<Matrix> forward(const std::vector<Matrix>& input) = 0;
  virtual std::vector<Matrix> backward(
      const std::vector<Matrix>& gradOutput) = 0;

  virtual MatrixSize getOutputSize() const = 0;
};

class ReLULayer : public Layer {
 private:
  // Store the last input for backward pass
  Matrix lastInput;
  MatrixSize inputSize;

 public:
  ReLULayer(const MatrixSize& inputSize) : inputSize(inputSize) {}

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
  MatrixSize getOutputSize() const override;
};

class ConvLayer : public Layer {
 private:
  size_t inputChannel;
  size_t outputChannel;
  MatrixSize inputSize;
  MatrixSize filterSize;
  int stride;

  // Filters (weights)
  std::vector<Matrix> filters;
  std::vector<Matrix> layerInput;
  std::vector<Matrix> layerOutput;

  std::vector<Matrix> filterGradient;

 public:
  ConvLayer(const size_t& inputChannel, const size_t& outputChannel,
            const MatrixSize& inputSize, const MatrixSize& filterSize,
            size_t stride);

  void initializeParameters();
  void randomInitialize(Matrix& matrix);
  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
  MatrixSize getOutputSize() const override;

 private:
  Matrix computeInputGradient(const Matrix& gradOutput);
  void updateParameters(const Matrix& gradOutput);
};

class DenseLayer : public Layer {
  // Similar structure to ConvLayer, but for fully connected layers
};

#endif