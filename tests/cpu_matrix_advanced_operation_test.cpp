#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "MatLib/Matrix.h"

class CPUMatrixAdvancedOperationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common test matrices
    mat3x3 = CPU::Matrix(3, 3);
    mat3x3(0, 0) = 1;
    mat3x3(0, 1) = 2;
    mat3x3(0, 2) = 3;
    mat3x3(1, 0) = 4;
    mat3x3(1, 1) = 5;
    mat3x3(1, 2) = 6;
    mat3x3(2, 0) = 7;
    mat3x3(2, 1) = 8;
    mat3x3(2, 2) = 9;

    mat2x3 = CPU::Matrix(2, 3);
    mat2x3(0, 0) = 1;
    mat2x3(0, 1) = 2;
    mat2x3(0, 2) = 3;
    mat2x3(1, 0) = 4;
    mat2x3(1, 1) = 5;
    mat2x3(1, 2) = 6;
  }

  CPU::Matrix mat3x3;
  CPU::Matrix mat2x3;
};

// Test copy method
TEST_F(CPUMatrixAdvancedOperationTest, Copy) {
  CPU::Matrix copied = mat3x3.copy();

  EXPECT_EQ(copied.m(), mat3x3.m());
  EXPECT_EQ(copied.n(), mat3x3.n());

  for (size_t i = 0; i < mat3x3.m(); ++i) {
    for (size_t j = 0; j < mat3x3.n(); ++j) {
      EXPECT_EQ(copied(i, j), mat3x3(i, j));
    }
  }

  // Verify it's a deep copy
  copied(0, 0) = 999;
  EXPECT_NE(copied(0, 0), mat3x3(0, 0));
}

// Test reshape method
TEST_F(CPUMatrixAdvancedOperationTest, Reshape) {
  CPU::Matrix mat = CPU::Matrix(2, 6);
  for (size_t i = 0; i < 12; ++i) {
    mat(i / 6, i % 6) = i + 1;
  }

  // Reshape to 3x4
  mat.reshape(3, 4);
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 4);

  // Check data integrity (row-major order)
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(0, 3), 4);
  EXPECT_EQ(mat(1, 0), 5);
  EXPECT_EQ(mat(2, 3), 12);

  // Test single parameter reshape
  mat.reshape(12);
  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), 12);

  // Test MatrixSize reshape
  mat.reshape(MatrixSize(4, 3));
  EXPECT_EQ(mat.m(), 4);
  EXPECT_EQ(mat.n(), 3);
}

// Test resize method
TEST_F(CPUMatrixAdvancedOperationTest, Resize) {
  CPU::Matrix mat = CPU::Matrix(2, 2);
  mat(0, 0) = 1;
  mat(0, 1) = 2;
  mat(1, 0) = 3;
  mat(1, 1) = 4;

  // Resize to larger
  mat.resize(3, 3);
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 3);
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(0, 1), 2);
  EXPECT_EQ(mat(0, 2), 3);
  EXPECT_EQ(mat(1, 0), 4);

  // Resize to smaller
  mat.resize(2, 2);
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 2);

  // Test single parameter resize
  mat.resize(4);
  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), 4);

  // Test MatrixSize resize
  mat.resize(MatrixSize(2, 3));
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 3);
}

// Test fill method
TEST_F(CPUMatrixAdvancedOperationTest, Fill) {
  CPU::Matrix mat(2, 3);
  mat.fill(7.5f);

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(mat(i, j), 7.5f);
    }
  }
}

// Test ones method
TEST_F(CPUMatrixAdvancedOperationTest, Ones) {
  CPU::Matrix mat;

  // Test ones with dimensions
  mat.ones(2, 3);
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(mat(i, j), 1.0f);
    }
  }

  // Test ones with single parameter
  mat.ones(5);
  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), 5);
  for (size_t j = 0; j < 5; ++j) {
    EXPECT_EQ(mat(0, j), 1.0f);
  }

  // Test ones with MatrixSize
  mat.ones(MatrixSize(3, 2));
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 2);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(mat(i, j), 1.0f);
    }
  }

  // Test ones on existing CPU::Matrix
  CPU::Matrix existing(2, 2);
  existing.fill(5.0f);
  existing.ones();
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(existing(i, j), 1.0f);
    }
  }
}

// Test zeros method
TEST_F(CPUMatrixAdvancedOperationTest, Zeros) {
  CPU::Matrix mat;

  // Test zeros with dimensions
  mat.zeros(2, 3);
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(mat(i, j), 0.0f);
    }
  }

  // Test zeros with single parameter
  mat.zeros(4);
  EXPECT_EQ(mat.m(), 1);
  EXPECT_EQ(mat.n(), 4);
  for (size_t j = 0; j < 4; ++j) {
    EXPECT_EQ(mat(0, j), 0.0f);
  }

  // Test zeros with MatrixSize
  mat.zeros(MatrixSize(3, 2));
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 2);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(mat(i, j), 0.0f);
    }
  }

  // Test zeros on existing CPU::Matrix
  CPU::Matrix existing(2, 2);
  existing.fill(5.0f);
  existing.zeros();
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(existing(i, j), 0.0f);
    }
  }
}

// Test arange method
TEST_F(CPUMatrixAdvancedOperationTest, Arange) {
  CPU::Matrix mat;

  // Test arange with end only
  mat.arange(5.0f);
  EXPECT_EQ(mat.n(), 5);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mat(0, i), static_cast<float>(i));
  }

  // Test arange with start and end
  mat.arange(2.0f, 7.0f);
  EXPECT_EQ(mat.n(), 5);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mat(0, i), static_cast<float>(i + 2));
  }

  // Test arange with start, end, and step
  mat.arange(1.0f, 10.0f, 2.0f);
  EXPECT_EQ(mat.n(), 5);  // 1, 3, 5, 7, 9
  EXPECT_EQ(mat(0, 0), 1.0f);
  EXPECT_EQ(mat(0, 1), 3.0f);
  EXPECT_EQ(mat(0, 2), 5.0f);
  EXPECT_EQ(mat(0, 3), 7.0f);
  EXPECT_EQ(mat(0, 4), 9.0f);
}

// Test rand method
TEST_F(CPUMatrixAdvancedOperationTest, Rand) {
  CPU::Matrix mat(3, 3);
  mat.rand(-1.0f, 1.0f);

  // Check that all values are within range
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_GE(mat(i, j), -1.0f);
      EXPECT_LE(mat(i, j), 1.0f);
    }
  }

  // Test with positive range
  mat.rand(0.0f, 10.0f);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_GE(mat(i, j), 0.0f);
      EXPECT_LE(mat(i, j), 10.0f);
    }
  }
}

// Test identity CPU::Matrix methods (I, identity, eye)
TEST_F(CPUMatrixAdvancedOperationTest, IdentityMatrix) {
  CPU::Matrix mat;

  // Test I method
  mat.I(3);
  EXPECT_EQ(mat.m(), 3);
  EXPECT_EQ(mat.n(), 3);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_EQ(mat(i, j), 1.0f);
      } else {
        EXPECT_EQ(mat(i, j), 0.0f);
      }
    }
  }

  // Test identity method (alias for I)
  CPU::Matrix mat2;
  mat2.identity(4);
  EXPECT_EQ(mat2.m(), 4);
  EXPECT_EQ(mat2.n(), 4);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      if (i == j) {
        EXPECT_EQ(mat2(i, j), 1.0f);
      } else {
        EXPECT_EQ(mat2(i, j), 0.0f);
      }
    }
  }

  // Test eye method (alias for I)
  CPU::Matrix mat3;
  mat3.eye(2);
  EXPECT_EQ(mat3.m(), 2);
  EXPECT_EQ(mat3.n(), 2);
  EXPECT_EQ(mat3(0, 0), 1.0f);
  EXPECT_EQ(mat3(0, 1), 0.0f);
  EXPECT_EQ(mat3(1, 0), 0.0f);
  EXPECT_EQ(mat3(1, 1), 1.0f);
}

// Test transpose methods (T, transpose)
TEST_F(CPUMatrixAdvancedOperationTest, Transpose) {
  CPU::Matrix original = mat2x3.copy();

  // Test T method
  mat2x3.T();
  EXPECT_EQ(mat2x3.m(), 3);
  EXPECT_EQ(mat2x3.n(), 2);

  // Check transposed values
  for (size_t i = 0; i < original.m(); ++i) {
    for (size_t j = 0; j < original.n(); ++j) {
      EXPECT_EQ(mat2x3(j, i), original(i, j));
    }
  }

  // Test transpose method (alias for T)
  CPU::Matrix mat = CPU::Matrix(3, 2);
  mat(0, 0) = 1;
  mat(0, 1) = 2;
  mat(1, 0) = 3;
  mat(1, 1) = 4;
  mat(2, 0) = 5;
  mat(2, 1) = 6;

  mat.transpose();
  EXPECT_EQ(mat.m(), 2);
  EXPECT_EQ(mat.n(), 3);
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(0, 1), 3);
  EXPECT_EQ(mat(0, 2), 5);
  EXPECT_EQ(mat(1, 0), 2);
  EXPECT_EQ(mat(1, 1), 4);
  EXPECT_EQ(mat(1, 2), 6);
}

// Test flip method
TEST_F(CPUMatrixAdvancedOperationTest, Flip) {
  CPU::Matrix mat = mat3x3.copy();

  // Test row flip (axis = 0)
  mat.flip(0);
  EXPECT_EQ(mat(0, 0), 7);
  EXPECT_EQ(mat(0, 1), 8);
  EXPECT_EQ(mat(0, 2), 9);
  EXPECT_EQ(mat(1, 0), 4);
  EXPECT_EQ(mat(1, 1), 5);
  EXPECT_EQ(mat(1, 2), 6);
  EXPECT_EQ(mat(2, 0), 1);
  EXPECT_EQ(mat(2, 1), 2);
  EXPECT_EQ(mat(2, 2), 3);

  // Test column flip (axis = 1)
  mat = mat3x3.copy();
  mat.flip(1);
  EXPECT_EQ(mat(0, 0), 3);
  EXPECT_EQ(mat(0, 1), 2);
  EXPECT_EQ(mat(0, 2), 1);
  EXPECT_EQ(mat(1, 0), 6);
  EXPECT_EQ(mat(1, 1), 5);
  EXPECT_EQ(mat(1, 2), 4);
  EXPECT_EQ(mat(2, 0), 9);
  EXPECT_EQ(mat(2, 1), 8);
  EXPECT_EQ(mat(2, 2), 7);
}

// Test rotate90 method
TEST_F(CPUMatrixAdvancedOperationTest, Rotate90) {
  CPU::Matrix mat = mat3x3.copy();

  // Test rotate90 k=1 (90 degrees counter-clockwise)
  mat.rotate90(1);
  EXPECT_EQ(mat(0, 0), 3);
  EXPECT_EQ(mat(0, 1), 6);
  EXPECT_EQ(mat(0, 2), 9);
  EXPECT_EQ(mat(1, 0), 2);
  EXPECT_EQ(mat(1, 1), 5);
  EXPECT_EQ(mat(1, 2), 8);
  EXPECT_EQ(mat(2, 0), 1);
  EXPECT_EQ(mat(2, 1), 4);
  EXPECT_EQ(mat(2, 2), 7);

  // Test rotate90 k=2 (180 degrees)
  mat = mat3x3.copy();
  mat.rotate90(2);
  EXPECT_EQ(mat(0, 0), 9);
  EXPECT_EQ(mat(0, 1), 8);
  EXPECT_EQ(mat(0, 2), 7);
  EXPECT_EQ(mat(1, 0), 6);
  EXPECT_EQ(mat(1, 1), 5);
  EXPECT_EQ(mat(1, 2), 4);
  EXPECT_EQ(mat(2, 0), 3);
  EXPECT_EQ(mat(2, 1), 2);
  EXPECT_EQ(mat(2, 2), 1);

  // Test rotate90 k=3 (90 degrees clockwise)
  mat = mat3x3.copy();
  mat.rotate90(3);
  EXPECT_EQ(mat(0, 0), 7);
  EXPECT_EQ(mat(0, 1), 4);
  EXPECT_EQ(mat(0, 2), 1);
  EXPECT_EQ(mat(1, 0), 8);
  EXPECT_EQ(mat(1, 1), 5);
  EXPECT_EQ(mat(1, 2), 2);
  EXPECT_EQ(mat(2, 0), 9);
  EXPECT_EQ(mat(2, 1), 6);
  EXPECT_EQ(mat(2, 2), 3);

  // Test rotate90 k=4 (360 degrees, should be same as original)
  mat = mat3x3.copy();
  mat.rotate90(4);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(mat(i, j), mat3x3(i, j));
    }
  }
}

// Test abs and fabs methods
TEST_F(CPUMatrixAdvancedOperationTest, AbsoluteMethods) {
  CPU::Matrix mat(2, 2);
  mat(0, 0) = -1.5f;
  mat(0, 1) = 2.5f;
  mat(1, 0) = -3.0f;
  mat(1, 1) = 4.0f;

  // Test abs method
  mat.abs();
  EXPECT_EQ(mat(0, 0), 1.5f);
  EXPECT_EQ(mat(0, 1), 2.5f);
  EXPECT_EQ(mat(1, 0), 3.0f);
  EXPECT_EQ(mat(1, 1), 4.0f);

  // Test fabs method (alias for abs)
  CPU::Matrix mat2(2, 2);
  mat2(0, 0) = -2.5f;
  mat2(0, 1) = 3.5f;
  mat2(1, 0) = -4.0f;
  mat2(1, 1) = 5.0f;

  mat2.fabs();
  EXPECT_EQ(mat2(0, 0), 2.5f);
  EXPECT_EQ(mat2(0, 1), 3.5f);
  EXPECT_EQ(mat2(1, 0), 4.0f);
  EXPECT_EQ(mat2(1, 1), 5.0f);
}

// Test max, min, sum methods
TEST_F(CPUMatrixAdvancedOperationTest, AggregationMethods) {
  CPU::Matrix mat(2, 3);
  mat(0, 0) = 1.0f;
  mat(0, 1) = 5.0f;
  mat(0, 2) = 3.0f;
  mat(1, 0) = 2.0f;
  mat(1, 1) = 4.0f;
  mat(1, 2) = 6.0f;

  EXPECT_EQ(mat.max(), 6.0f);
  EXPECT_EQ(mat.min(), 1.0f);
  EXPECT_EQ(mat.sum(), 21.0f);
}

// Test column operations
TEST_F(CPUMatrixAdvancedOperationTest, ColumnOperations) {
  // Test single column
  CPU::Matrix col = mat3x3.col(1);
  EXPECT_EQ(col.m(), 3);
  EXPECT_EQ(col.n(), 1);
  EXPECT_EQ(col(0, 0), 2);
  EXPECT_EQ(col(1, 0), 5);
  EXPECT_EQ(col(2, 0), 8);

  // Test multiple columns
  CPU::Matrix cols = mat3x3.cols(0, 2);
  EXPECT_EQ(cols.m(), 3);
  EXPECT_EQ(cols.n(), 2);
  EXPECT_EQ(cols(0, 0), 1);
  EXPECT_EQ(cols(0, 1), 2);
  EXPECT_EQ(cols(1, 0), 4);
  EXPECT_EQ(cols(1, 1), 5);
  EXPECT_EQ(cols(2, 0), 7);
  EXPECT_EQ(cols(2, 1), 8);
}

// Test row operations
TEST_F(CPUMatrixAdvancedOperationTest, RowOperations) {
  // Test single row
  CPU::Matrix row = mat3x3.row(1);
  EXPECT_EQ(row.m(), 1);
  EXPECT_EQ(row.n(), 3);
  EXPECT_EQ(row(0, 0), 4);
  EXPECT_EQ(row(0, 1), 5);
  EXPECT_EQ(row(0, 2), 6);

  // Test multiple rows
  CPU::Matrix rows = mat3x3.rows(0, 2);
  EXPECT_EQ(rows.m(), 2);
  EXPECT_EQ(rows.n(), 3);
  EXPECT_EQ(rows(0, 0), 1);
  EXPECT_EQ(rows(0, 1), 2);
  EXPECT_EQ(rows(0, 2), 3);
  EXPECT_EQ(rows(1, 0), 4);
  EXPECT_EQ(rows(1, 1), 5);
  EXPECT_EQ(rows(1, 2), 6);
}

// Test submatrix
TEST_F(CPUMatrixAdvancedOperationTest, Submatrix) {
  CPU::Matrix sub = mat3x3.submatrix(0, 2, 1, 3);
  EXPECT_EQ(sub.m(), 2);
  EXPECT_EQ(sub.n(), 2);
  EXPECT_EQ(sub(0, 0), 2);
  EXPECT_EQ(sub(0, 1), 3);
  EXPECT_EQ(sub(1, 0), 5);
  EXPECT_EQ(sub(1, 1), 6);
}

// Test concatenate
TEST_F(CPUMatrixAdvancedOperationTest, Concatenate) {
  CPU::Matrix mat1(2, 2);
  mat1(0, 0) = 1;
  mat1(0, 1) = 2;
  mat1(1, 0) = 3;
  mat1(1, 1) = 4;

  CPU::Matrix mat2(2, 2);
  mat2(0, 0) = 5;
  mat2(0, 1) = 6;
  mat2(1, 0) = 7;
  mat2(1, 1) = 8;

  // Concatenate along axis 0 (rows)
  CPU::Matrix result1 = mat1.copy();
  result1.concatenate(mat2, 0);
  EXPECT_EQ(result1.m(), 4);
  EXPECT_EQ(result1.n(), 2);
  EXPECT_EQ(result1(0, 0), 1);
  EXPECT_EQ(result1(0, 1), 2);
  EXPECT_EQ(result1(1, 0), 3);
  EXPECT_EQ(result1(1, 1), 4);
  EXPECT_EQ(result1(2, 0), 5);
  EXPECT_EQ(result1(2, 1), 6);
  EXPECT_EQ(result1(3, 0), 7);
  EXPECT_EQ(result1(3, 1), 8);

  // Concatenate along axis 1 (columns)
  CPU::Matrix result2 = mat1.copy();
  result2.concatenate(mat2, 1);
  EXPECT_EQ(result2.m(), 2);
  EXPECT_EQ(result2.n(), 4);
  EXPECT_EQ(result2(0, 0), 1);
  EXPECT_EQ(result2(0, 1), 2);
  EXPECT_EQ(result2(0, 2), 5);
  EXPECT_EQ(result2(0, 3), 6);
  EXPECT_EQ(result2(1, 0), 3);
  EXPECT_EQ(result2(1, 1), 4);
  EXPECT_EQ(result2(1, 2), 7);
  EXPECT_EQ(result2(1, 3), 8);
}

// Test dot product
TEST_F(CPUMatrixAdvancedOperationTest, DotProduct) {
  CPU::Matrix mat1(2, 3);
  mat1(0, 0) = 1;
  mat1(0, 1) = 2;
  mat1(0, 2) = 3;
  mat1(1, 0) = 4;
  mat1(1, 1) = 5;
  mat1(1, 2) = 6;

  CPU::Matrix mat2(3, 2);
  mat2(0, 0) = 7;
  mat2(0, 1) = 8;
  mat2(1, 0) = 9;
  mat2(1, 1) = 10;
  mat2(2, 0) = 11;
  mat2(2, 1) = 12;

  mat1.dot(mat2);
  EXPECT_EQ(mat1.m(), 2);
  EXPECT_EQ(mat1.n(), 2);

  // Expected results: [58, 64; 139, 154]
  EXPECT_EQ(mat1(0, 0), 58);   // 1*7 + 2*9 + 3*11
  EXPECT_EQ(mat1(0, 1), 64);   // 1*8 + 2*10 + 3*12
  EXPECT_EQ(mat1(1, 0), 139);  // 4*7 + 5*9 + 6*11
  EXPECT_EQ(mat1(1, 1), 154);  // 4*8 + 5*10 + 6*12
}

// Test convolution (basic)
TEST_F(CPUMatrixAdvancedOperationTest, Convolution) {
  CPU::Matrix input(4, 4);
  // Create a simple input pattern
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      input(i, j) = i * 4 + j + 1;
    }
  }

  CPU::Matrix kernel(2, 2);
  kernel(0, 0) = 1;
  kernel(0, 1) = 0;
  kernel(1, 0) = 0;
  kernel(1, 1) = 1;

  // Test basic convolution
  input.convolution(kernel);
  EXPECT_EQ(input.m(), 3);
  EXPECT_EQ(input.n(), 3);

  // XXX Needs implementation
  /*
    // Test convolution with stride
    CPU::Matrix input2(4, 4);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        input2(i, j) = i * 4 + j + 1;
      }
    }
    input2.convolution(kernel, 2);
    EXPECT_EQ(input2.m(), 2);
    EXPECT_EQ(input2.n(), 2);

    // Test convolution with padding
    CPU::Matrix input3(3, 3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        input3(i, j) = i * 3 + j + 1;
      }
    }
    input3.convolution(kernel, 1, CPU::Matrix::PaddingType::FULL);
    EXPECT_EQ(input3.m(), 4);
    EXPECT_EQ(input3.n(), 4);
  */
}

// Test maxPooling
TEST_F(CPUMatrixAdvancedOperationTest, MaxPooling) {
  CPU::Matrix input(4, 4);
  input(0, 0) = 1;
  input(0, 1) = 3;
  input(0, 2) = 2;
  input(0, 3) = 4;
  input(1, 0) = 5;
  input(1, 1) = 6;
  input(1, 2) = 7;
  input(1, 3) = 8;
  input(2, 0) = 1;
  input(2, 1) = 2;
  input(2, 2) = 3;
  input(2, 3) = 4;
  input(3, 0) = 5;
  input(3, 1) = 6;
  input(3, 2) = 7;
  input(3, 3) = 8;

  // Test max pooling with 2x2 kernel and stride 2
  input.maxPooling(2);
  EXPECT_EQ(input.m(), 2);
  EXPECT_EQ(input.n(), 2);
  EXPECT_EQ(input(0, 0), 6);  // Max of top-left quadrant
  EXPECT_EQ(input(0, 1), 8);  // Max of top-right quadrant
  EXPECT_EQ(input(1, 0), 6);  // Max of bottom-left quadrant
  EXPECT_EQ(input(1, 1), 8);  // Max of bottom-right quadrant

  // XXX Needs implementation for stride and padding
  /*
  // Test max pooling with padding
  CPU::Matrix input2(5, 5);
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      input2(i, j) = i * 5 + j + 1;
    }
  }

  input2.maxPooling(3);
  EXPECT_EQ(input2.m(), 5);
  EXPECT_EQ(input2.n(), 5);
  */
}