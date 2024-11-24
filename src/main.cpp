#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <typeinfo>

#include "Matrix.h"
#include "MatrixTester.h"
#include "Matrix_CUDA.cuh"

#define MATRIXSIZE 2000

int main() {
  MatrixTester tester;

  Matrix mat1;
  Matrix mat2(MATRIXSIZE, MATRIXSIZE / 2);

  mat1.I(MATRIXSIZE);
  mat2.fill(2);

  tester.createTest("CPU Dot", "dot", mat1, mat2, false);
  tester.createTest("GPU Dot", "dot", mat1, mat2, true);

  tester.runTest("CPU Dot");
  tester.runTest("GPU Dot");

  tester.printTime("CPU Dot");
  tester.printTime("GPU Dot");

  tester.printError("CPU Dot", "GPU Dot");

  return 0;
}