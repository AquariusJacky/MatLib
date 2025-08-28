#include <chrono>
#include <iostream>

#include "MatLib/Matrix.h"

#define MATRIX_SIZE 8
#define FILL_VALUE 7.0f
#define SCALE_FACTOR 0.03f

int main() {
  // Create a CPU matrix
  CPU::Matrix cpuMatA(MATRIX_SIZE, MATRIX_SIZE);
  CPU::Matrix cpuMatB(MATRIX_SIZE, MATRIX_SIZE);
  cpuMatA.fill(13.0f);
  cpuMatB.fill(17.0f);

  // Create a GPU matrix using the CPU matrix
  GPU::Matrix gpuMatA(cpuMatA);
  GPU::Matrix gpuMatB(cpuMatB);

  ///////////////////////////////
  // Perform operations on GPU //
  ///////////////////////////////
  // gpuMatB.fill(FILL_VALUE);
  // gpuMatB.scale(SCALE_FACTOR);
  // gpuMatA.dot(gpuMatB);
  gpuMatA.add(gpuMatB);

  // Copy back to CPU

  CPU::Matrix cpuMatC;
  gpuMatB.toCPU(cpuMatC);
  // gpuMatB.toCPU(cpuMatB);

  std::cout << cpuMatC << std::endl;

  std::cout << "Matrix A after addition with Matrix B:" << std::endl;
  // std::cout << cpuMatA << std::endl;

  return 0;
}