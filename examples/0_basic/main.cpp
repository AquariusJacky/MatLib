#include <chrono>
#include <iostream>

#include "gpumatrix/CPUMatrix.h"
#include "gpumatrix/GPUMatrix.cuh"

#define MATRIX_SIZE 1000
#define FILL_VALUE 7.0f
#define SCALE_FACTOR 2.0f

int main() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
      transfer_to_gpu_time, calculation_time, transfer_to_cpu_time;

  // Create a CPU matrix
  CPU::Matrix cpuMat(MATRIX_SIZE, MATRIX_SIZE);
  start_time = std::chrono::high_resolution_clock::now();

  // Create a GPU matrix using the CPU matrix
  GPU::Matrix gpuMat(cpuMat);
  transfer_to_gpu_time = std::chrono::high_resolution_clock::now();

  ///////////////////////////////
  // Perform operations on GPU //
  ///////////////////////////////
  gpuMat.fill(FILL_VALUE);
  gpuMat.scale(SCALE_FACTOR);
  calculation_time = std::chrono::high_resolution_clock::now();

  // Copy back to CPU
  gpuMat.toCPU(cpuMat);
  transfer_to_cpu_time = std::chrono::high_resolution_clock::now();

  ////////////////////
  //                //
  // Output results //
  //                //
  ////////////////////

  for (size_t i = 0; i < MATRIX_SIZE; ++i) {
    for (size_t j = 0; j < MATRIX_SIZE; ++j) {
      if (cpuMat(i, j) != (FILL_VALUE * SCALE_FACTOR)) {
        std::cerr << "Error: cpuMat(" << i << ", " << j
                  << ") = " << cpuMat(i, j) << ", expected "
                  << (FILL_VALUE * SCALE_FACTOR) << std::endl;
        return 1;
      }
    }
  }

  std::cout << "Calculation:" << std::endl;
  std::cout << "  Fill " << FILL_VALUE << std::endl;
  std::cout << "  Scale " << SCALE_FACTOR << std::endl;
  std::cout << "Results Correct" << std::endl;

  std::cout << "\nTiming:" << std::endl;
  std::cout << "  Transfer to GPU:\t"
            << std::chrono::duration<float, std::milli>(transfer_to_gpu_time -
                                                        start_time)
                   .count()
            << " ms" << std::endl;
  std::cout << "  Calculation:    \t"
            << std::chrono::duration<float, std::milli>(calculation_time -
                                                        transfer_to_gpu_time)
                   .count()
            << " ms" << std::endl;
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