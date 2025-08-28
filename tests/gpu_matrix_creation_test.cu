#include <gtest/gtest.h>

#include "MatLib/Matrix.h"

class GPUMatrixCreationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA device
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    // Reset device at start
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    // Clear any existing errors
    cudaGetLastError();
  }

  void TearDown() override {
    // Wait for all operations to complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    // Check for any errors that occurred during the test
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }
};

// =============================================================================
// GPU Matrix Creation Tests
// =============================================================================

TEST_F(GPUMatrixCreationTest, GPUMatrixDefaultConstructor) {
  GPU::Matrix mat;
  EXPECT_EQ(mat.m(), 0);
  EXPECT_EQ(mat.n(), 0);
  EXPECT_EQ(mat.size(), 0);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixDimensionConstructor) {
  GPU::Matrix mat(3, 4);
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 4);
  EXPECT_EQ(mat.size(), 12);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixZeroDimensionConstructor) {
  GPU::Matrix mat1(0, 5);
  GPU::Matrix mat2(5, 0);
  GPU::Matrix mat3(0, 0);

  EXPECT_EQ(mat1.size(), 0);
  EXPECT_EQ(mat2.size(), 0);
  EXPECT_EQ(mat3.size(), 0);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixLargeDimensionConstructor) {
  GPU::Matrix mat(1000, 1000);
  EXPECT_EQ(mat.m(), 1000);
  EXPECT_EQ(mat.n(), 1000);
  EXPECT_EQ(mat.size(), 1000000);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixOutOfRangeDimensionConstructor) {
  EXPECT_THROW(GPU::Matrix mat(1000, 1001), std::runtime_error);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixCPUMatrixConstructor) {
  // Create a CPU matrix with known values
  CPU::Matrix cpu_mat(3, 2);
  // Assuming CPU::Matrix has a way to set values
  // You may need to adjust this based on your CPU::Matrix implementation

  GPU::Matrix gpu_mat(cpu_mat);
  EXPECT_EQ(gpu_mat.m(), 3);
  EXPECT_EQ(gpu_mat.n(), 2);
  EXPECT_EQ(gpu_mat.size(), 6);
}

TEST_F(GPUMatrixCreationTest, ToCPU) {
  CPU::Matrix cpu_mat(5, 4);
  cpu_mat.fill(3.14f);  // Fill with some values
  GPU::Matrix gpu_mat(cpu_mat);

  // Create CPU matrix to store result
  CPU::Matrix tocpu_mat;

  // Convert GPU matrix to CPU
  gpu_mat.toCPU(tocpu_mat);

  // Verify dimensions
  EXPECT_EQ(tocpu_mat.m(), 5);
  EXPECT_EQ(tocpu_mat.n(), 4);
  EXPECT_EQ(tocpu_mat.size(), 20);
  EXPECT_EQ(tocpu_mat.equal(cpu_mat), 1);  // Check if values match
}

TEST_F(GPUMatrixCreationTest, GPUMatrixCopyConstructor) {
  CPU::Matrix cpu_mat(4, 5);
  cpu_mat.fill(3.14f);  // Fill with some values

  GPU::Matrix original(cpu_mat);

  GPU::Matrix copy(original);
  EXPECT_EQ(copy.m(), 4);
  EXPECT_EQ(copy.n(), 5);
  EXPECT_EQ(copy.size(), 20);

  CPU::Matrix cpu_copy;
  copy.toCPU(cpu_copy);

  EXPECT_EQ(cpu_copy.m(), 4);
  EXPECT_EQ(cpu_copy.n(), 5);
  EXPECT_EQ(cpu_copy.size(), 20);
  EXPECT_EQ(cpu_copy.equal(cpu_mat), 1);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixAssignmentOperator) {
  CPU::Matrix cpu_mat(2, 3);
  cpu_mat.fill(1.5f);
  GPU::Matrix mat1(cpu_mat);

  GPU::Matrix mat2;
  mat2 = mat1;

  CPU::Matrix cpu_mat1, cpu_mat2;
  mat1.toCPU(cpu_mat1);
  mat2.toCPU(cpu_mat2);

  EXPECT_EQ(cpu_mat2.equal(cpu_mat1), 1);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixSelfAssignment) {
  GPU::Matrix mat(3, 3);
  mat.fill(2.0f);

  mat = mat;  // Self-assignment

  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 3);
  EXPECT_EQ(mat.size(), 9);
}

TEST_F(GPUMatrixCreationTest, GPUMatrixChainedAssignment) {
  CPU::Matrix cpu_mat(3, 2);
  cpu_mat.fill(1.0f);
  GPU::Matrix mat1(cpu_mat);

  GPU::Matrix mat2, mat3;
  mat3 = mat2 = mat1;

  CPU::Matrix cpu_mat2, cpu_mat3;
  mat2.toCPU(cpu_mat2);
  mat3.toCPU(cpu_mat3);

  EXPECT_EQ(cpu_mat2.equal(cpu_mat), 1);
  EXPECT_EQ(cpu_mat3.equal(cpu_mat), 1);
}