#ifndef CPUMATRIX_H
#define CPUMATRIX_H

#include <stdlib.h>

#include <iostream>
#include <vector>

namespace GPU {
class Matrix;  // class forward declaration
}

struct MatrixSize {
  size_t m;
  size_t n;

  MatrixSize() {
    m = 0;
    n = 0;
  }
  MatrixSize(size_t _n) {
    m = 1;
    n = _n;
  }
  MatrixSize(size_t _m, size_t _n) {
    m = _m;
    n = _n;
  }
  ~MatrixSize() {}
};

namespace CPU {

class Matrix {
 private:
  size_t m_;
  size_t n_;
  float* data_;  // Stores the actual data in row-major order
  size_t calculate_index(const size_t& x, const size_t& y) const;

 public:
  enum PaddingType { NONE, FULL };

 public:
  friend class GPU::Matrix;

  Matrix();
  Matrix(const size_t& n);
  Matrix(const size_t& m, const size_t& n);
  Matrix(const MatrixSize& size);
  Matrix(const Matrix& matB);
  Matrix(const std::vector<float>& vec);
  Matrix(const std::vector<std::vector<float>>& mat_from_vec);
  ~Matrix();

  size_t m() const { return m_; }
  size_t n() const { return n_; }
  size_t size() const { return m_ * n_; }

  float& operator()(const size_t& x, const size_t& y);
  const float& operator()(const size_t& x, const size_t& y) const;

  Matrix& operator=(const Matrix& matB);
  Matrix& operator+=(const Matrix& matB);
  Matrix& operator-=(const Matrix& matB);
  Matrix& operator*=(const float& scale);
  Matrix operator+(const Matrix& matB) const;
  Matrix operator-(const Matrix& matB) const;
  Matrix operator*(const float& scale) const;
  Matrix copy() const { return (*this); }

  Matrix& reshape(const size_t& n);
  Matrix& reshape(const size_t& m, const size_t& n);
  Matrix& reshape(const MatrixSize& size);
  Matrix& resize(const size_t& n);
  Matrix& resize(const size_t& m, const size_t& n);
  Matrix& resize(const MatrixSize& sz);

  Matrix& ones();
  Matrix& ones(const size_t& n);
  Matrix& ones(const size_t& m, const size_t& n);
  Matrix& ones(const MatrixSize& sz);
  Matrix& zeros();
  Matrix& zeros(const size_t& n);
  Matrix& zeros(const size_t& m, const size_t& n);
  Matrix& zeros(const MatrixSize& sz);

  Matrix& arange(const float& end);
  Matrix& arange(const float& start, const float& end);
  Matrix& arange(const float& start, const float& end, const float& step);
  Matrix& rand(const float& lower_limit, const float& upper_limit);

  Matrix& fill(const float& val);
  Matrix& I(const size_t& sz);
  Matrix& T();
  Matrix& flip(const size_t& axis);  // 0 for column, 1 for row
  Matrix& rotate90(const size_t& k);
  Matrix& identity(const size_t& sz) { return (*this).I(sz); }
  Matrix& eye(const size_t& n) { return (*this).I(n); }
  Matrix& transpose() { return (*this).T(); }
  Matrix& abs();                            // Absolute
  Matrix& fabs() { return (*this).abs(); }  // Floating point absolute
  // Only works with square matrix, matrix self dot
  Matrix& power(const float& power);
  float sum();

  Matrix col(const size_t& col_num);
  Matrix cols(const size_t& col_start, const size_t& col_end);
  Matrix row(const size_t& row_num);
  Matrix rows(const size_t& row_start, const size_t& row_end);
  Matrix submatrix(const size_t& row_start, const size_t& row_end,
                   const size_t& col_start, const size_t& col_end);

  Matrix& concatenate(const Matrix& matB, const size_t& axis);

  Matrix& dot(const Matrix& matB);
  Matrix& convolution(const Matrix& mask);
  Matrix& convolution(const Matrix& mask, const size_t& stride);
  Matrix& convolution(const Matrix& mask, const size_t& stride,
                      const PaddingType& padding_type);

  Matrix& maxPooling(const size_t& size);
};

std::ostream& operator<<(std::ostream&, const Matrix&);

}  // namespace CPU
#endif