#include "Layer.h"
#include "Matrix.h"

ConvLayer::ConvLayer(const size_t& inputChannel, const size_t& outputChannel,
                     const MatrixSize& inputSize, const MatrixSize& filterSize,
                     size_t stride = 1)
    : inputSize(inputSize),
      filterSize(filterSize),
      inputChannel(inputChannel),
      outputChannel(outputChannel),
      stride(stride) {
  // Initialize filters and biases with random values
  initializeParameters();
}

void ConvLayer::initializeParameters() {
  float rand_limit;

  // Initialize filters with random values
  for (int i = 0; i < outputChannel; ++i) {
    filters.push_back(Matrix(filterSize.m, filterSize.n));

    // May use some initialization strategy (e.g., Xavier/Glorot)
    filters[i].rand(rand_limit * -1, rand_limit);
  }
  layerOutput = std::vector<Matrix>(outputChannel);
}

std::vector<Matrix> ConvLayer::forward(const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }
  if (input[0].m() != inputSize.m || input[0].n() != inputSize.n) {
    throw std::runtime_error("Input size incorrect in ConvLayer forward");
  }

  // Store input for backward pass
  layerInput = input;

  // Apply each filter to the input
  for (size_t channel = 0; channel < inputChannel; channel++) {
    for (size_t filter_num = 0; filter_num < outputChannel; filter_num++) {
      size_t curr_output = channel * inputChannel + filter_num;
      layerOutput[curr_output] = input[channel];
      layerOutput[curr_output].convolution(filters[filter_num]);
    }
  }

  return layerOutput;
}

std::vector<Matrix> ConvLayer::backward(const std::vector<Matrix>& gradOutput) {
  // Gradient with respect to inputs
  std::vector<Matrix> inputGradients(
      layerInput.size(), Matrix(layerInput[0].m(), layerInput[0].n()));

  // Gradient with respect to filters
  std::vector<Matrix> filterGradients(outputChannel,
                                      Matrix(filterSize.m, filterSize.n));

  // Track the index of gradOutputs
  int gradOutputIndex = 0;

  // Process each input
  for (size_t inputIdx = 0; inputIdx < layerInput.size(); ++inputIdx) {
    // Process each filter for this input
    for (int f = 0; f < outputChannel; ++f) {
      Matrix& currentInput = layerInput[inputIdx];
      Matrix& currentGradOutput = gradOutputs[gradOutputIndex];

      // Compute filter gradient
      Matrix filterGradient = currentInput;
      filterGradient.convolution(currentGradOutput);
      filterGradients[f] += filterGradient;

      // Compute input gradient
      Matrix rotatedFilter = filters[f];
      rotatedFilter.rotate90(2);  // rotate 180

      Matrix inputDeltaForFilter = currentGradOutput;
      inputDeltaForFilter.convolution(rotatedFilter);

      // Accumulate input gradient
      inputGradients[inputIdx] += inputDeltaForFilter;

      // Move to next gradient output
      gradOutputIndex++;
    }
  }

  // Update filters
  for (int f = 0; f < outputChannel; ++f) {
    filters[f] -= learningRate * filterGradients[f];
  }

  return inputGradients;
}

MatrixSize ConvLayer::getOutputSize() const {
  // Calculate output size based on input, filter, and stride
  int outM = (inputSize.m - filterSize.m) / stride + 1;
  int outN = (inputSize.n - filterSize.n) / stride + 1;
  return MatrixSize(outM, outN);
}

Matrix ConvLayer::computeInputGradient(const Matrix& gradOutput) {
  // Compute gradient with respect to input
  // Placeholder - replace with actual gradient computation
  return Matrix();
}

void ConvLayer::updateParameters(const Matrix& gradOutput) {
  // Update filter and bias parameters using gradient descent
  // Placeholder - replace with actual parameter update logic
}