#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <typeinfo>

#include "Matrix.h"
#include "MatrixTester.h"
#include "Matrix_CUDA.cuh"

#define MATRIXSIZE 5

int main() {
  MatrixTester tester;

  Matrix mat1, mat2;

  // mat1.arange(MATRIXSIZE * MATRIXSIZE);
  // mat1.reshape(MATRIXSIZE, MATRIXSIZE);
  mat1.I(MATRIXSIZE);
  mat2.I(3);

  // mat1.concatenate(mat2, 1);

  // Matrix mat3;
  // mat3.arange(36).reshape(6, 6);
  // std::cout << mat3;

  // mat3.maxPooling(4);
  // std::cout << mat3;

  tester.createTest("CPU Convolution", "convolution", mat1, mat2, false);

  tester.runTest("CPU Convolution");
  tester.printTime("CPU Convolution");
  tester.printResult("CPU Convolution");

  // tester.createTest("CPU MaxPooling", "maxPooling", mat1, 3, false);
  // tester.createTest("GPU MaxPooling", "maxPooling", mat1, 3, true);

  // tester.runTest("GPU MaxPooling");
  // tester.printTime("GPU MaxPooling");

  // tester.runTest("CPU MaxPooling");
  // tester.printTime("CPU MaxPooling");

  // tester.printError("CPU MaxPooling", "GPU MaxPooling");

  return 0;
}