#include <iostream>

#include "MatLib/Matrix.h"
#include "MatrixTester.h"

#define MATRIX_SIZE 2000

int main() {
  MatrixTester tester;
  CPU::Matrix matA(MATRIX_SIZE, MATRIX_SIZE);
  CPU::Matrix matB(MATRIX_SIZE, MATRIX_SIZE);

  tester.createTest("Matrix A Ones CPU", "ones", matA, false);
  tester.runTest("Matrix A Ones CPU");

  tester.createTest("Matrix A Ones GPU", "ones", matA, true);
  tester.runTest("Matrix A Ones GPU");

  tester.printTime("Matrix A Ones CPU");
  tester.printTime("Matrix A Ones GPU");
  tester.printError("Matrix A Ones CPU", "Matrix A Ones GPU");

  std::cout << "=========================================" << std::endl;

  matB.fill(7);

  tester.createTest("Matrix Add CPU", "add", matA, matB, false);
  tester.runTest("Matrix Add CPU");

  tester.createTest("Matrix Add GPU", "add", matA, matB, true);
  tester.runTest("Matrix Add GPU");

  tester.printTime("Matrix Add CPU");
  tester.printTime("Matrix Add GPU");
  tester.printError("Matrix Add CPU", "Matrix Add GPU");

  std::cout << "=========================================" << std::endl;

  tester.createTest("Matrix Dot CPU", "dot", matA, matB, false);
  tester.runTest("Matrix Dot CPU");

  tester.createTest("Matrix Dot GPU", "dot", matA, matB, true);
  tester.runTest("Matrix Dot GPU");

  tester.printTime("Matrix Dot CPU");
  tester.printTime("Matrix Dot GPU");
  tester.printError("Matrix Dot CPU", "Matrix Dot GPU");

  std::cout << "=========================================" << std::endl;

  CPU::Matrix matC(5, 5);
  matC.rand(0, 1);

  tester.createTest("Matrix Conv CPU", "conv", matA, matC, false);
  tester.runTest("Matrix Conv CPU");

  tester.createTest("Matrix Conv GPU", "conv", matA, matC, true);
  tester.runTest("Matrix Conv GPU");

  tester.printTime("Matrix Conv CPU");
  tester.printTime("Matrix Conv GPU");
  tester.printError("Matrix Conv CPU", "Matrix Conv GPU");

  std::cout << "=========================================" << std::endl;

  return 0;
}