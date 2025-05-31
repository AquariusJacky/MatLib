#include <iostream>

#include "MatrixTester.h"
#include "gpumatrix/CPUMatrix.h"
#include "gpumatrix/GPUMatrix.cuh"

int main() {
  MatrixTester tester;
  CPUMatrix cpumatA(3, 3);
  CPUMatrix cpumatB(3, 3);

  tester.createTest("Matrix A Fill CUDA", "ones", cpumatA, true);
  tester.runTest("Matrix A Fill CUDA");
  tester.printResult("Matrix A Fill CUDA");

  tester.createTest("Matrix A Fill CUDA", "ones", cpumatA, true);

  return 0;
}