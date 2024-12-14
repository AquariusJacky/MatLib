#include "Layer.h"
#include "Matrix.h"

std::vector<Matrix> ReLULayer::forward(const std::vector<Matrix>& input) {
  // Store input for backward pass
  lastInput = input;

  // Create output matrix of same size
  Matrix output(input.m(), input.n());

  // Apply ReLU: max(0, x)
  for (int i = 0; i < input.m(); ++i) {
    for (int j = 0; j < input.n(); ++j) {
      output(i, j) = std::max(0.0, input(i, j));
    }
  }

  return output;
}

std::vector<Matrix> ReLULayer::backward(const std::vector<Matrix>& gradOutput) {
  // Compute gradient for input
  Matrix inputGradient(lastInput.m(), lastInput.n());

  // ReLU gradient:
  // - If input > 0, gradient is 1 (pass through)
  // - If input <= 0, gradient is 0 (kill gradient)
  for (int i = 0; i < lastInput.m(); ++i) {
    for (int j = 0; j < lastInput.n(); ++j) {
      inputGradient(i, j) = (lastInput(i, j) > 0) ? gradOutput(i, j) : 0.0;
    }
  }

  return inputGradient;
}

MatrixSize ReLULayer::getOutputSize() const {
  // ReLU doesn't change the input size
  return inputSize;
}