#include <gtest/gtest.h>

#include "MatLib/Matrix.h"

class GPUMatrixAdvancedOperationTest : public ::testing::Test {
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
// GPU Matrix Advanced Operation Tests
// =============================================================================

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixDotProduct) {
  GPU::Matrix mat1(3, 4);
  GPU::Matrix mat2(4, 2);

  mat1.fill(2.0f);
  mat2.fill(3.0f);

  mat1.dot(mat2);

  EXPECT_EQ(mat1.m(), 3);
  EXPECT_EQ(mat1.n(), 2);

  // Expected result: each element should be 2*3*4 = 24
  GPU::Matrix expected(3, 2);
  expected.fill(24.0f);

  EXPECT_EQ(mat1.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixDotProductSquareMatrices) {
  GPU::Matrix mat1(3, 3);
  GPU::Matrix mat2(3, 3);

  mat1.fill(1.0f);
  mat2.fill(2.0f);

  mat1.dot(mat2);

  EXPECT_EQ(mat1.m(), 3);
  EXPECT_EQ(mat1.n(), 3);

  // Expected: each element should be 1*2*3 = 6
  GPU::Matrix expected(3, 3);
  expected.fill(6.0f);

  EXPECT_EQ(mat1.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixDotProductWithZero) {
  GPU::Matrix mat1(2, 3);
  GPU::Matrix mat2(3, 2);

  mat1.fill(5.0f);
  mat2.fill(0.0f);

  mat1.dot(mat2);

  GPU::Matrix zero_result(2, 2);
  zero_result.fill(0.0f);

  EXPECT_EQ(mat1.equal(zero_result), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixMaxReduction) {
  GPU::Matrix mat(4, 4);
  mat.fill(3.0f);

  GPU::Matrix result = mat.max();

  // Max of all 3.0f should be 3.0f
  GPU::Matrix expected(1, 1);
  expected.fill(3.0f);

  EXPECT_EQ(result.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixMinReduction) {
  GPU::Matrix mat(3, 5);
  mat.fill(7.5f);

  GPU::Matrix result = mat.min();

  GPU::Matrix expected(1, 1);
  expected.fill(7.5f);

  EXPECT_EQ(result.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixSumReduction) {
  GPU::Matrix mat(2, 3);
  mat.fill(4.0f);

  GPU::Matrix result = mat.sum();

  // Sum of 6 elements, each 4.0f = 24.0f
  GPU::Matrix expected(1, 1);
  expected.fill(24.0f);

  EXPECT_EQ(result.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixSumReductionZero) {
  GPU::Matrix mat(5, 5);
  mat.fill(0.0f);

  GPU::Matrix result = mat.sum();

  GPU::Matrix expected(1, 1);
  expected.fill(0.0f);

  EXPECT_EQ(result.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixReductionEmptyMatrix) {
  GPU::Matrix mat(0, 0);

  EXPECT_THROW(mat.max(), std::runtime_error);
  EXPECT_THROW(mat.min(), std::runtime_error);
  EXPECT_THROW(mat.sum(), std::runtime_error);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixConvolution) {
  GPU::Matrix mat(5, 5);
  GPU::Matrix mask(3, 3);

  mat.fill(1.0f);
  mask.fill(1.0f);

  mat.convolution(mask);

  // After convolution with 3x3 mask of ones on 5x5 matrix of ones,
  // the output should be smaller and contain specific values
  // The exact size depends on your convolution implementation (valid, same,
  // full)
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixConvolutionIdentityMask) {
  GPU::Matrix mat(4, 4);
  GPU::Matrix identity_mask(1, 1);

  mat.fill(5.0f);
  identity_mask.fill(1.0f);

  GPU::Matrix original(mat);
  mat.convolution(identity_mask);

  // Convolution with 1x1 mask of value 1 should preserve the original
  EXPECT_EQ(mat.equal(original), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixMaxPooling) {
  GPU::Matrix mat(4, 4);
  mat.fill(8.0f);

  mat.maxPooling(2);

  // Max pooling with size 2 on 4x4 should result in 2x2
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 2);

  // All values should still be 8.0f
  GPU::Matrix expected(2, 2);
  expected.fill(8.0f);

  EXPECT_EQ(mat.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixMaxPoolingLargeSize) {
  GPU::Matrix mat(6, 6);
  mat.fill(3.0f);

  mat.maxPooling(3);

  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 2);

  GPU::Matrix expected(2, 2);
  expected.fill(3.0f);

  EXPECT_EQ(mat.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixCPUTransfer) {
  GPU::Matrix gpu_mat(3, 2);
  gpu_mat.fill(4.5f);

  CPU::Matrix cpu_mat(3, 2);
  gpu_mat.toCPU(cpu_mat);

  // Create another GPU matrix from CPU matrix to verify transfer
  GPU::Matrix gpu_mat2(cpu_mat);

  EXPECT_EQ(gpu_mat.equal(gpu_mat2), 1);
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixLargeMatrixOperations) {
  GPU::Matrix large_mat(1000, 1000);
  large_mat.fill(1.0f);
  large_mat.scale(2.0f);

  GPU::Matrix expected(1000, 1000);
  expected.fill(2.0f);

  EXPECT_EQ(large_mat.equal(expected), 1);
}

TEST_F(GPUMatrixAdvancedOperationTest, GPUMatrixChainedOperations) {
  GPU::Matrix mat(3, 3);
  mat.fill(2.0f);

  GPU::Matrix add_mat(3, 3);
  add_mat.fill(3.0f);

  // Chain operations
  mat.add(add_mat).scale(2.0f);

  GPU::Matrix expected(3, 3);
  expected.fill(10.0f);  // (2+3)*2 = 10

  EXPECT_EQ(mat.equal(expected), 1);
}