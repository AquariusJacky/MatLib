#include "Layer.h"
#include "Matrix.h"

ConvLayer::ConvLayer(const size_t& numInput, const size_t& numWeight,
                     const MatrixSize& inputSize, const MatrixSize& weightSize,
                     size_t stride = 1)
    : inputSize(inputSize),
      weightSize(weightSize),
      numInput(numInput),
      numWeight(numWeight),
      stride(stride) {
  // Initialize weights and biases with random values
  init();
}

void ConvLayer::init() {
  float rand_limit = 10000;

  // Initialize weights with random values
  for (size_t i = 0; i < numWeight; ++i) {
    weights.push_back(Matrix(weightSize.m, weightSize.n));

    // May use some initialization strategy (e.g., Xavier/Glorot)
    weights[i].rand(rand_limit * -1, rand_limit);
  }
  layerOutput = std::vector<Matrix>(numInput * numWeight);

  inputGradient = std::vector<Matrix>(numInput);
  weightGradient = std::vector<Matrix>(numWeight);
}

std::vector<Matrix> ConvLayer::forward(const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }
  if (input[0].m() != inputSize.m || input[0].n() != inputSize.n) {
    throw std::runtime_error("Input size incorrect for ConvLayer forward");
  }

  // Store input for backward pass
  layerInput = input;

  // Apply each weight to the input
  for (size_t channel = 0; channel < numInput; channel++) {
    for (size_t weight_num = 0; weight_num < numWeight; weight_num++) {
      size_t curr_output = numInput * channel + weight_num;
      layerOutput[curr_output] = input[channel];
      layerOutput[curr_output].convolution(weights[weight_num], stride);
    }
  }

  return layerOutput;
}

std::vector<Matrix> ConvLayer::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != numInput) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  for (size_t channel = 0; channel < numInput; channel++) {
    for (size_t weight_num = 0; weight_num < numWeight; weight_num++) {
      size_t curr_output = channel * numInput + weight_num;
      Matrix curr_weightGradient = layerInput[channel];
      Matrix curr_inputGradient = weights[weight_num];

      // Weight gradient = Conv(input, outputGradient)
      curr_weightGradient.convolution(outputGradient[curr_output], stride);

      // This is for previous layer
      // Input gradient = Conv(weight rotate 180, outputGradient)
      curr_inputGradient.rotate90(2).convolution(
          outputGradient[curr_output], stride, Matrix::PaddingType::FULL);

      weightGradient[weight_num] += curr_weightGradient;
      inputGradient[channel] += curr_inputGradient;
    }
  }

  // Update weights
  updateWeight();

  return inputGradient;
}

MatrixSize ConvLayer::getOutputSize() const {
  // Calculate output size based on input, weight, and stride
  size_t outM = (inputSize.m - weightSize.m) / stride + 1;
  size_t outN = (inputSize.n - weightSize.n) / stride + 1;
  return MatrixSize(outM, outN);
}

void ConvLayer::updateWeight() {
  // Update weight and bias parameters using gradient descent
  // Placeholder - replace with actual parameter update logic
}