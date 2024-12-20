#include "Layer.h"
#include "Matrix.h"

ReLULayer::ReLULayer(const size_t& numInput, const MatrixSize& inputSize)
    : numInput(numInput), inputSize(inputSize) {
  init();
}

void ReLULayer::init() {
  layerOutput = std::vector<Matrix>(numInput);
  inputGradient = std::vector<Matrix>(numInput);
}

std::vector<Matrix> ReLULayer::forward(const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }
  if (input[0].m() != inputSize.m || input[0].n() != inputSize.n) {
    throw std::runtime_error("Input size incorrect for ConvLayer forward");
  }

  for (int channel = 0; channel < numInput; channel++) {
    layerOutput[channel] = layerInput[channel];
    for (int i = 0; i < inputSize.m; ++i) {
      for (int j = 0; j < inputSize.n; ++j) {
        if (layerOutput[channel](i, j) < 0) {
          layerOutput[channel](i, j) = 0;
        }
      }
    }
  }

  // Store input for backward pass
  layerInput = input;

  return layerOutput;
}

std::vector<Matrix> ReLULayer::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != numInput) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  // ReLU gradient:
  // - If input > 0, gradient is 1 (pass through)
  // - If input <= 0, gradient is 0 (kill gradient)
  for (int channel = 0; channel < numInput; channel++) {
    inputGradient[channel] = outputGradient[channel];
    for (int i = 0; i < inputSize.m; ++i) {
      for (int j = 0; j < inputSize.n; ++j) {
        if (layerInput[channel](i, j) < 0) {
          inputGradient[channel](i, j) = 0;
        }
      }
    }
  }

  return inputGradient;
}

MatrixSize ReLULayer::getOutputSize() const { return inputSize; }