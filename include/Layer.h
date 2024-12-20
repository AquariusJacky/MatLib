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
  size_t numInput;
  MatrixSize inputSize;

  // Weights (weights)
  std::vector<Matrix> layerInput;
  std::vector<Matrix> layerOutput;

  std::vector<Matrix> inputGradient;

 public:
  ReLULayer(const size_t& numInput, const MatrixSize& inputSize);

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
  MatrixSize getOutputSize() const override;

 private:
  void init();
};

class ConvLayer : public Layer {
 private:
  size_t numInput;
  size_t numWeight;
  MatrixSize inputSize;
  MatrixSize weightSize;
  size_t stride;

  // Weights (weights)
  std::vector<Matrix> weights;
  std::vector<Matrix> layerInput;
  std::vector<Matrix> layerOutput;

  std::vector<Matrix> weightGradient;
  std::vector<Matrix> inputGradient;

 public:
  ConvLayer(const size_t& numInput, const size_t& num_weight,
            const MatrixSize& inputSize, const MatrixSize& weightSize,
            size_t stride);

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
  MatrixSize getOutputSize() const override;

 private:
  void init();
  void updateWeight();
};

// Fully connected layer (FCN)
class DenseLayer : public Layer {
 private:
  MatrixSize inputSize;
  MatrixSize outputSize;

  // Weights (weights)
  std::vector<Matrix> weight;
  std::vector<Matrix> bias;
  std::vector<Matrix> layerInput;
  std::vector<Matrix> layerOutput;

  std::vector<Matrix> weightGradient;
  std::vector<Matrix> biasGradient;
  std::vector<Matrix> inputGradient;

 public:
  DenseLayer(const MatrixSize& inputSize, const MatrixSize& outputSize);
  DenseLayer(const size_t& inputSize, const size_t& outputSize);

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
  MatrixSize getOutputSize() const override;

 private:
  void init();
  void updateWeight();
};

#endif