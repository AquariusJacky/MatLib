#include "Layer.h"
#include "Matrix.h"

DenseLayer::DenseLayer(const MatrixSize& inputSize,
                       const MatrixSize& outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
  // Initialize weight and biases with random values
  init();
}

DenseLayer::DenseLayer(const size_t& inputSize, const size_t& outputSize)
    : inputSize(MatrixSize(inputSize, 1)),
      outputSize(MatrixSize(outputSize, 1)) {
  // Initialize weight and biases with random values
  init();
}

void DenseLayer::init() {
  if (inputSize.n != 1) {
    throw std::runtime_error("Dense layer: Input must have size m x 1");
  }
  float rand_limit = 10000;

  weight.push_back(Matrix(inputSize.m, outputSize.m));
  // May use some initialization strategy (e.g., Xavier/Glorot)
  weight[0].rand(rand_limit * -1, rand_limit);

  layerOutput = std::vector<Matrix>(1);

  inputGradient = std::vector<Matrix>(1);
  weightGradient = std::vector<Matrix>(1);
}

std::vector<Matrix> DenseLayer::forward(const std::vector<Matrix>& input) {
  if (input.size() != 1) {
    throw std::runtime_error("Dense forward: Input must only contain 1 matrix");
  }
  if (input[0].m() != inputSize.m || input[0].n() != 1) {
    throw std::runtime_error("Input size incorrect for DenseLayer forward");
  }

  // Store input for backward pass
  layerInput = input;

  // Apply each weight to the input
  layerOutput[0] = input[0];
  layerOutput[0].dot(weight[0]);
  layerOutput[0] += bias[0];

  return layerOutput;
}

std::vector<Matrix> DenseLayer::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != 1) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  Matrix curr_weightGradient = outputGradient[0];
  Matrix curr_biasGradient = layerInput[0];
  Matrix curr_inputGradient = weight[0];

  // Weight gradient = dot(outputGradient, input^T)
  curr_weightGradient.dot(layerInput[0].copy().T());
  // Bias gradient = dot(outputGradient, input^T)
  curr_weightGradient.dot(layerInput[0].copy().T());

  // This is for previous layer
  // Input gradient = dot(weight^T, outputGradient)
  curr_inputGradient.T().dot(outputGradient[0]);

  weightGradient[0] += curr_weightGradient;
  inputGradient[0] += curr_inputGradient;

  // Update weight
  updateWeight();

  return inputGradient;
}

MatrixSize DenseLayer::getOutputSize() const { return outputSize; }

void DenseLayer::updateWeight() {
  // Update weight and bias parameters using gradient descent
  // Placeholder - replace with actual parameter update logic
}