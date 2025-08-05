#include <chrono>
#include <iostream>

#include "MatLib/Matrix.h"

#define MATRIX_SIZE 8
#define FILL_VALUE 7.0f
#define SCALE_FACTOR 0.03f

int comparison(float val, float golden) {
  if (val != golden) {
    std::cerr << "Error: Max mismatch. Expected " << golden << ", got " << val
              << std::endl;
    std::cout << "============== FAIL ==============" << std::endl;
    return 1;
  }

  std::cout << "============== PASS ==============" << std::endl;
  return 0;
}

int main() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
      transfer_to_gpu_time, calculation_time, transfer_to_cpu_time;

  // Create a CPU matrix
  CPU::Matrix cpuMatA(MATRIX_SIZE, MATRIX_SIZE);
  CPU::Matrix cpuMatB(MATRIX_SIZE, MATRIX_SIZE / 2);
  cpuMatA.fill(13.0f);
  std::cout << "Matrix A:" << std::endl;
  std::cout << cpuMatA << std::endl;

  CPU::Matrix golden = cpuMatA.copy();

  start_time = std::chrono::high_resolution_clock::now();

  // Create a GPU matrix using the CPU matrix
  GPU::Matrix gpuMatA(cpuMatA);
  GPU::Matrix gpuMatB(cpuMatB);
  transfer_to_gpu_time = std::chrono::high_resolution_clock::now();

  ///////////////////////////////
  // Perform operations on GPU //
  ///////////////////////////////
  gpuMatB.fill(FILL_VALUE);
  gpuMatB.scale(SCALE_FACTOR);
  gpuMatA.dot(gpuMatB);
  calculation_time = std::chrono::high_resolution_clock::now();

  // // Copy back to CPU
  // gpuMat.toCPU(CMat);

  gpuMatA.toCPU(cpuMatA);
  gpuMatB.toCPU(cpuMatB);
  transfer_to_cpu_time = std::chrono::high_resolution_clock::now();

  ////////////////////
  // Output results //
  ////////////////////

  golden.dot(cpuMatB);
  std::cout << cpuMatB << std::endl;

  std::cout << "Matrix A after dot product with Matrix B:" << std::endl;
  std::cout << cpuMatA << std::endl;

  for (size_t i = 0; i < golden.m(); i++) {
    for (size_t j = 0; j < golden.n(); j++) {
      if (cpuMatA(i, j) != golden(i, j)) {
        std::cerr << "Error: Mismatch at (" << i << ", " << j << "). Expected "
                  << golden(i, j) << ", got " << cpuMatA(i, j) << std::endl;
        std::cout << "============== FAIL ==============" << std::endl;
      }
    }
  }
  std::cout << "============== PASS ==============" << std::endl;

  ////////////////
  // Print time //
  ////////////////
  std::cout << "\nTime:" << std::endl;
  std::cout << "  Transfer to GPU:\t"
            << std::chrono::duration<float, std::milli>(transfer_to_gpu_time -
                                                        start_time)
                   .count()
            << " ms" << std::endl;
  std::cout << "  Calculation:    \t"
            << std::chrono::duration<float, std::milli>(calculation_time -
                                                        transfer_to_gpu_time)
                   .count()
            << " ms\n"
            << "    fill\n"
            << "    scale\n"
            << "    dot\n";
  std::cout << "  Transfer to CPU:\t"
            << std::chrono::duration<float, std::milli>(transfer_to_cpu_time -
                                                        calculation_time)
                   .count()
            << " ms" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "  Total time:     \t"
            << std::chrono::duration<float, std::milli>(transfer_to_cpu_time -
                                                        start_time)
                   .count()
            << " ms" << std::endl;

  return 0;
}