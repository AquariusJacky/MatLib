#include <iostream>

#include "MatrixTester.h"
#include "MatLib/Matrix.h"

int main() {
  MatrixTester tester;
  CPU::Matrix cpumatA(3, 3);

  tester.createTest("Matrix A Fill CUDA", "ones", cpumatA, true);
  tester.runTest("Matrix A Fill CUDA");
  tester.printResult("Matrix A Fill CUDA");

  return 0;
}