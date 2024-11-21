#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <typeinfo>

#include "Matrix.h"
#include "MatrixTester.h"
#include "Matrix_CUDA.cuh"

#define MATRIXSIZE 1000

int main() {
  MatrixTester tester;

  Matrix mat1(MATRIXSIZE, MATRIXSIZE);
  Matrix mat2(MATRIXSIZE, MATRIXSIZE / 2);
  tester.createTest("Ones 10000", "ones", mat1);
  tester.runTest("Ones 10000");
  tester.printTime("Ones 10000");

  mat1.I(MATRIXSIZE);
  mat2.fill(2);

  tester.createTest("Dot 10000", "dot", mat1, mat2);
  tester.runTest("Dot 10000");
  tester.printTime("Dot 10000");

  return 0;
}