#include <gtest/gtest.h>

#include <vector>

#include "MatLib/Matrix.h"

class CPUMatrixBasicOperationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    small_rows = 3;
    small_cols = 4;
    large_rows = 1000;
    large_cols = 500;
  }

  void TearDown() override {}

  int single_row;
  int small_rows, small_cols;
  int large_rows, large_cols;
};

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixEqualOperator) {
  std::vector<float> vec1D = {1.0f, 2.0f, 3.0f, 4.0f};
  CPU::Matrix matA(vec1D);
  CPU::Matrix matB;

  matB = matA;  // Copy assignment

  EXPECT_EQ(matB.m(), matA.m());
  EXPECT_EQ(matB.n(), matA.n());

  for (size_t i = 0; i < vec1D.size(); i++) {
    EXPECT_EQ(matB(0, i), matA(0, i));
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixPlusOperator) {
  // Test 1D vector addition
  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};

  CPU::Matrix matA(vec1);
  CPU::Matrix matB(vec2);
  CPU::Matrix result = matA + matB;

  EXPECT_EQ(result.m(), matA.m());
  EXPECT_EQ(result.n(), matA.n());

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result(0, i), vec1[i] + vec2[i]);
  }

  // Test 2D matrix addition
  std::vector<std::vector<float>> mat1 = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  std::vector<std::vector<float>> mat2 = {
      {2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}};
  std::vector<std::vector<float>> expected2D = {
      {3.0f, 5.0f}, {7.0f, 9.0f}, {11.0f, 13.0f}};

  CPU::Matrix matC(mat1);
  CPU::Matrix matD(mat2);
  CPU::Matrix result2D = matC + matD;

  EXPECT_EQ(result2D.m(), 3);
  EXPECT_EQ(result2D.n(), 2);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(result2D(i, j), expected2D[i][j]);
    }
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixMinusOperator) {
  // Test 1D vector subtraction
  std::vector<float> vec1 = {10.0f, 8.0f, 6.0f, 4.0f};
  std::vector<float> vec2 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected = {9.0f, 6.0f, 3.0f, 0.0f};

  CPU::Matrix matA(vec1);
  CPU::Matrix matB(vec2);
  CPU::Matrix result = matA - matB;

  EXPECT_EQ(result.m(), matA.m());
  EXPECT_EQ(result.n(), matA.n());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(result(0, i), expected[i]);
  }

  // Test 2D matrix subtraction
  std::vector<std::vector<float>> mat1 = {
      {10.0f, 9.0f}, {8.0f, 7.0f}, {6.0f, 5.0f}};
  std::vector<std::vector<float>> mat2 = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  std::vector<std::vector<float>> expected2D = {
      {9.0f, 7.0f}, {5.0f, 3.0f}, {1.0f, -1.0f}};

  CPU::Matrix matC(mat1);
  CPU::Matrix matD(mat2);
  CPU::Matrix result2D = matC - matD;

  EXPECT_EQ(result2D.m(), 3);
  EXPECT_EQ(result2D.n(), 2);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(result2D(i, j), expected2D[i][j]);
    }
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixScalarMultiplicationOperator) {
  // Test scalar multiplication with 1D vector
  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  float scalar = 2.5f;
  std::vector<float> expected = {2.5f, 5.0f, 7.5f, 10.0f};

  CPU::Matrix matA(vec1);
  CPU::Matrix result = matA * scalar;

  EXPECT_EQ(result.m(), matA.m());
  EXPECT_EQ(result.n(), matA.n());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(result(0, i), expected[i]);
  }

  // Test scalar multiplication with 2D matrix
  std::vector<std::vector<float>> mat1 = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  float scalar2 = 3.0f;
  std::vector<std::vector<float>> expected2D = {
      {3.0f, 6.0f}, {9.0f, 12.0f}, {15.0f, 18.0f}};

  CPU::Matrix matB(mat1);
  CPU::Matrix result2D = matB * scalar2;

  EXPECT_EQ(result2D.m(), 3);
  EXPECT_EQ(result2D.n(), 2);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(result2D(i, j), expected2D[i][j]);
    }
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixPlusEqualOperator) {
  // Test += with 1D vectors
  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};

  CPU::Matrix matA(vec1);
  CPU::Matrix matB(vec2);

  matA += matB;

  EXPECT_EQ(matA.m(), 1);
  EXPECT_EQ(matA.n(), 4);

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(matA(0, i), expected[i]);
  }

  // Test += with 2D matrices
  std::vector<std::vector<float>> mat1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  std::vector<std::vector<float>> mat2 = {{2.0f, 3.0f}, {4.0f, 5.0f}};
  std::vector<std::vector<float>> expected2D = {{3.0f, 5.0f}, {7.0f, 9.0f}};

  CPU::Matrix matC(mat1);
  CPU::Matrix matD(mat2);

  matC += matD;

  EXPECT_EQ(matC.m(), 2);
  EXPECT_EQ(matC.n(), 2);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(matC(i, j), expected2D[i][j]);
    }
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixMinusEqualOperator) {
  // Test -= with 1D vectors
  std::vector<float> vec1 = {10.0f, 8.0f, 6.0f, 4.0f};
  std::vector<float> vec2 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected = {9.0f, 6.0f, 3.0f, 0.0f};

  CPU::Matrix matA(vec1);
  CPU::Matrix matB(vec2);

  matA -= matB;

  EXPECT_EQ(matA.m(), 1);
  EXPECT_EQ(matA.n(), 4);

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(matA(0, i), expected[i]);
  }

  // Test -= with 2D matrices
  std::vector<std::vector<float>> mat1 = {{10.0f, 9.0f}, {8.0f, 7.0f}};
  std::vector<std::vector<float>> mat2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  std::vector<std::vector<float>> expected2D = {{9.0f, 7.0f}, {5.0f, 3.0f}};

  CPU::Matrix matC(mat1);
  CPU::Matrix matD(mat2);

  matC -= matD;

  EXPECT_EQ(matC.m(), 2);
  EXPECT_EQ(matC.n(), 2);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(matC(i, j), expected2D[i][j]);
    }
  }
}

TEST_F(CPUMatrixBasicOperationTest,
       CPUMatrixScalarMultiplicationEqualOperator) {
  // Test *= with scalar on 1D vector
  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  float scalar = 2.5f;
  std::vector<float> expected = {2.5f, 5.0f, 7.5f, 10.0f};

  CPU::Matrix matA(vec1);

  matA *= scalar;

  EXPECT_EQ(matA.m(), 1);
  EXPECT_EQ(matA.n(), 4);

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(matA(0, i), expected[i]);
  }

  // Test *= with scalar on 2D matrix
  std::vector<std::vector<float>> mat1 = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  float scalar2 = 3.0f;
  std::vector<std::vector<float>> expected2D = {
      {3.0f, 6.0f}, {9.0f, 12.0f}, {15.0f, 18.0f}};

  CPU::Matrix matB(mat1);

  matB *= scalar2;

  EXPECT_EQ(matB.m(), 3);
  EXPECT_EQ(matB.n(), 2);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(matB(i, j), expected2D[i][j]);
    }
  }
}

// Edge cases and error handling tests
TEST_F(CPUMatrixBasicOperationTest, CPUMatrixOperationsEdgeCases) {
  // Test operations with zero matrices
  std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> ones = {1.0f, 1.0f, 1.0f, 1.0f};

  CPU::Matrix zeroMat(zeros);
  CPU::Matrix onesMat(ones);

  // Addition with zero
  CPU::Matrix result1 = onesMat + zeroMat;
  for (size_t i = 0; i < ones.size(); i++) {
    EXPECT_FLOAT_EQ(result1(0, i), 1.0f);
  }

  // Subtraction with zero
  CPU::Matrix result2 = onesMat - zeroMat;
  for (size_t i = 0; i < ones.size(); i++) {
    EXPECT_FLOAT_EQ(result2(0, i), 1.0f);
  }

  // Multiplication by zero
  CPU::Matrix result3 = onesMat * 0.0f;
  for (size_t i = 0; i < ones.size(); i++) {
    EXPECT_FLOAT_EQ(result3(0, i), 0.0f);
  }

  // Multiplication by one
  CPU::Matrix result4 = onesMat * 1.0f;
  for (size_t i = 0; i < ones.size(); i++) {
    EXPECT_FLOAT_EQ(result4(0, i), 1.0f);
  }
}

TEST_F(CPUMatrixBasicOperationTest, CPUMatrixEqual) {
  std::vector<float> vec1 = {10.0f, 8.0f, 6.0f, 4.0f};
  std::vector<float> vec2 = {1.0f, 2.0f, 3.0f, 4.0f};

  CPU::Matrix magA(vec1);
  CPU::Matrix matB(vec1);
  CPU::Matrix matC(vec2);

  EXPECT_EQ(magA.equal(matB), 1);  // Should be equal
  EXPECT_EQ(magA.equal(matC), 0);  // Should not be equal
}