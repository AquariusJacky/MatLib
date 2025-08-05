#include <gtest/gtest.h>

#include <vector>

#include "MatLib/Matrix.h"

class GPUMatrixCreationTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(GPUMatrixCreationTest, GPUMatrix_DefaultConstructor) {}