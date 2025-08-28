#include <gtest/gtest.h>

#include "MatLib/Matrix.h"

class GPUMatrixBasicOperationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    cudaDeviceSynchronize();  // Wait for any pending operations
  }

  void TearDown() override {
    cudaDeviceSynchronize();
    cudaGetLastError();  // Clear any pending errors
  }
};

// =============================================================================
// GPU Matrix Basic Operation Tests
// =============================================================================

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixFillOperation) {
  GPU::Matrix mat(3, 4);
  mat.fill(5.0f);

  // Create a CPU matrix to verify the fill operation
  CPU::Matrix cpu_mat(3, 4);
  mat.toCPU(cpu_mat);

  for (size_t i = 0; i < cpu_mat.m(); ++i) {
    for (size_t j = 0; j < cpu_mat.n(); ++j) {
      EXPECT_FLOAT_EQ(cpu_mat(i, j), 5.0f);
    }
  }
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixFillZero) {
  GPU::Matrix mat(5, 5);
  mat.fill(0.0f);

  CPU::Matrix cpu_mat(5, 5);
  mat.toCPU(cpu_mat);
  
  for (size_t i = 0; i < cpu_mat.m(); ++i) {
    for (size_t j = 0; j < cpu_mat.n(); ++j) {
      EXPECT_FLOAT_EQ(cpu_mat(i, j), 0.0f);
    }
  }
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixFillNegative) {
  GPU::Matrix mat(2, 3);
  mat.fill(-2.5f);

  CPU::Matrix cpu_mat(2, 3);
  mat.toCPU(cpu_mat);
  
  for (size_t i = 0; i < cpu_mat.m(); ++i) {
    for (size_t j = 0; j < cpu_mat.n(); ++j) {
      EXPECT_FLOAT_EQ(cpu_mat(i, j), -2.5f);
    }
  }
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixAddOperation) {
  CPU::Matrix cpu_mat1(3, 3);

  cpu_mat1.fill(2.0f);
  GPU::Matrix mat1(cpu_mat1);

  cpu_mat1.fill(3.0f);
  GPU::Matrix mat2(cpu_mat1);

  // mat1.fill(2.0f);
  // mat2.fill(3.0f);

  mat1.add(mat2);

  // // Create expected result
  // GPU::Matrix expected(3, 3);
  // expected.fill(5.0f);

  // EXPECT_EQ(mat1.equal(expected), 1);

  mat1.toCPU(cpu_mat1);
  // Verify the result
  for (size_t i = 0; i < cpu_mat1.m(); ++i) {
    for (size_t j = 0; j < cpu_mat1.n(); ++j) {
      EXPECT_FLOAT_EQ(cpu_mat1(i, j), 5.0f);
    }
  }
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixAddZero) {
  GPU::Matrix mat1(4, 4);
  GPU::Matrix zero_mat(4, 4);

  mat1.fill(7.0f);
  zero_mat.fill(0.0f);

  GPU::Matrix original(mat1);
  mat1.add(zero_mat);

  EXPECT_EQ(mat1.equal(original), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixAddNegative) {
  GPU::Matrix mat1(2, 2);
  GPU::Matrix mat2(2, 2);

  mat1.fill(5.0f);
  mat2.fill(-2.0f);

  mat1.add(mat2);

  GPU::Matrix expected(2, 2);
  expected.fill(3.0f);

  EXPECT_EQ(mat1.equal(expected), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixScaleOperation) {
  GPU::Matrix mat(3, 2);
  mat.fill(4.0f);
  mat.scale(2.5f);

  GPU::Matrix expected(3, 2);
  expected.fill(10.0f);

  EXPECT_EQ(mat.equal(expected), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixScaleZero) {
  GPU::Matrix mat(3, 3);
  mat.fill(5.0f);
  mat.scale(0.0f);

  GPU::Matrix zero_mat(3, 3);
  zero_mat.fill(0.0f);

  EXPECT_EQ(mat.equal(zero_mat), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixScaleNegative) {
  GPU::Matrix mat(2, 4);
  mat.fill(3.0f);
  mat.scale(-2.0f);

  GPU::Matrix expected(2, 4);
  expected.fill(-6.0f);

  EXPECT_EQ(mat.equal(expected), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixScaleOne) {
  GPU::Matrix mat(4, 4);
  mat.fill(7.5f);

  GPU::Matrix original(mat);
  mat.scale(1.0f);

  EXPECT_EQ(mat.equal(original), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixEqualOperation) {
  GPU::Matrix mat1(3, 3);
  GPU::Matrix mat2(3, 3);

  mat1.fill(2.0f);
  mat2.fill(2.0f);

  EXPECT_EQ(mat1.equal(mat2), 1);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixNotEqual) {
  GPU::Matrix mat1(3, 3);
  GPU::Matrix mat2(3, 3);

  mat1.fill(2.0f);
  mat2.fill(3.0f);

  EXPECT_EQ(mat1.equal(mat2), 0);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixEqualDifferentDimensions) {
  GPU::Matrix mat1(3, 3);
  GPU::Matrix mat2(3, 4);

  mat1.fill(1.0f);
  mat2.fill(1.0f);

  EXPECT_EQ(mat1.equal(mat2), 0);
}

TEST_F(GPUMatrixBasicOperationTest, GPUMatrixEqualEmptyMatrices) {
  GPU::Matrix mat1;
  GPU::Matrix mat2;

  EXPECT_EQ(mat1.equal(mat2), 1);
}