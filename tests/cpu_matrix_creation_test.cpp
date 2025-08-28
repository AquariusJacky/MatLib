#include <gtest/gtest.h>

#include <vector>

#include "MatLib/Matrix.h"

class CPUMatrixCreationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    small_rows = 3;
    small_cols = 4;
    large_rows = 1000;
    large_cols = 1000;

    for (int i = 0; i < large_cols; ++i) {
      vec1D.push_back(static_cast<float>(i));
    }
    for (int i = 0; i < large_cols; ++i) {
      for (int j = 0; j < small_rows; ++j) {
        vec2D.push_back(std::vector<float>());
        vec2D[j].push_back(static_cast<float>(i + j));
      }
    }
  }

  void TearDown() override {}

  int single_row;
  int small_rows, small_cols;
  int large_rows, large_cols;
  std::vector<float> vec1D;
  std::vector<std::vector<float>> vec2D;
};

// CPU Matrix Creation Tests
TEST_F(CPUMatrixCreationTest, CPUMatrixDefaultConstructor) {
  CPU::Matrix mat;  // Default constructor

  EXPECT_EQ(mat.m(), 0);
  EXPECT_EQ(mat.n(), 0);
  EXPECT_EQ(mat.size(), 0);

  EXPECT_THROW(mat(0, 0), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_0_size) {
  CPU::Matrix mat(0);

  EXPECT_EQ(mat.m(), 0);
  EXPECT_EQ(mat.n(), 0);
  EXPECT_EQ(mat.size(), 0);

  EXPECT_THROW(mat(0, 0), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_Small_1D) {
  // 1D matrix acts like vector: 1 x N
  CPU::Matrix mat(small_cols);

  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), small_cols);
  EXPECT_EQ(mat.size(), small_cols);

  for (int j = 0; j < small_cols; j++) {
    EXPECT_EQ(mat(0, j), 0.0f);  // Default initialization
  }
  EXPECT_THROW(mat(1, 0), std::runtime_error);
  EXPECT_THROW(mat(0, small_cols), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_Large_1D) {
  // 1D matrix acts like vector: 1 x N
  CPU::Matrix mat(large_cols);

  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), large_cols);
  EXPECT_EQ(mat.size(), large_cols);

  for (int j = 0; j < large_cols; j++) {
    EXPECT_EQ(mat(0, j), 0.0f);  // Default initialization
  }
  EXPECT_THROW(mat(1, 0), std::runtime_error);
  EXPECT_THROW(mat(0, large_cols), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_OutOfBounds_1D) {
  EXPECT_THROW(CPU::Matrix mat(1000001), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_Small_2D) {
  CPU::Matrix mat(small_rows, small_cols);

  EXPECT_EQ(mat.m(), small_rows);
  EXPECT_EQ(mat.n(), small_cols);
  EXPECT_EQ(mat.size(), small_rows * small_cols);

  for (int i = 0; i < small_rows; ++i) {
    for (int j = 0; j < small_cols; ++j) {
      EXPECT_EQ(mat(i, j), 0.0f);  // Default initialization
    }
  }
  EXPECT_THROW(mat(small_rows, small_cols), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_Large_2D) {
  CPU::Matrix mat(large_rows, large_cols);

  EXPECT_EQ(mat.m(), large_rows);
  EXPECT_EQ(mat.n(), large_cols);
  EXPECT_EQ(mat.size(), large_rows * large_cols);

  for (int i = 0; i < large_rows; ++i) {
    for (int j = 0; j < large_cols; ++j) {
      EXPECT_EQ(mat(i, j), 0.0f);  // Default initialization
    }
  }
  EXPECT_THROW(mat(large_rows, large_cols), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_OutOfBounds_2D) {
  EXPECT_THROW(CPU::Matrix mat(1000, 1001), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixConstructor_SingleElement) {
  CPU::Matrix mat(1, 1);
  // Test edge case of 1 x 1 matrix
  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), 1);

  EXPECT_EQ(mat(0, 0), 0.0f);
  EXPECT_THROW(mat(0, 1), std::runtime_error);
  EXPECT_THROW(mat(1, 0), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixVectorConstructor_1D) {
  std::vector<float> vec1D = {1.0f, 2.0f, 3.0f, 4.0f};
  CPU::Matrix mat(vec1D);

  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), vec1D.size());

  for (size_t i = 0; i < vec1D.size(); i++) {
    EXPECT_EQ(mat(0, i), vec1D[i]);
  }
  EXPECT_THROW(mat(0, vec1D.size()), std::runtime_error);
  EXPECT_THROW(mat(vec1D.size(), 0), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixVectorConstructor_2D) {
  std::vector<std::vector<float>> vec2D = {{1.0f, 2.0f, 3.0f},
                                           {4.0f, 5.0f, 6.0f}};
  CPU::Matrix mat(vec2D);

  EXPECT_EQ(mat.m(), vec2D.size());
  EXPECT_EQ(mat.n(), vec2D[0].size());

  for (size_t i = 0; i < vec2D.size(); i++) {
    for (size_t j = 0; j < vec2D[i].size(); j++) {
      EXPECT_EQ(mat(i, j), vec2D[i][j]);
    }
  }
  EXPECT_THROW(mat(vec2D.size(), 0), std::runtime_error);
  EXPECT_THROW(mat(0, vec2D[0].size()), std::runtime_error);
}

TEST_F(CPUMatrixCreationTest, CPUMatrixCopyConstructor) {
  std::vector<std::vector<float>> vec2D = {{1.0f, 2.0f, 3.0f},
                                           {4.0f, 5.0f, 6.0f}};
  CPU::Matrix matA(vec2D);

  CPU::Matrix matB(matA);  // Copy constructor

  EXPECT_EQ(matB.m(), vec2D.size());
  EXPECT_EQ(matB.n(), vec2D[0].size());

  for (size_t i = 0; i < vec2D.size(); i++) {
    for (size_t j = 0; j < vec2D[i].size(); j++) {
      EXPECT_EQ(matB(i, j), vec2D[i][j]);
    }
  }

  EXPECT_THROW(matB(vec2D.size(), 0), std::runtime_error);
  EXPECT_THROW(matB(0, vec2D[0].size()), std::runtime_error);
}