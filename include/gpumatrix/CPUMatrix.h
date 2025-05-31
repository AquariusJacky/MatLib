#ifndef CPUMATRIX_H
#define CPUMATRIX_H

#include <stdlib.h>

#include <iostream>
#include <vector>

class GPUMatrix;  // class forward declaration

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

class CPUMatrix {
 private:
  size_t m_;
  size_t n_;
  float* data_;  // Stores the actual data in row-major order
  size_t calculate_index(const size_t& x, const size_t& y) const;

 public:
  enum PaddingType { NONE, FULL };

 public:
  friend class GPUMatrix;

  CPUMatrix();
  CPUMatrix(const size_t& n);
  CPUMatrix(const size_t& m, const size_t& n);
  CPUMatrix(const MatrixSize& size);
  CPUMatrix(const CPUMatrix& matB);
  CPUMatrix(const std::vector<float>& vec);
  CPUMatrix(const std::vector<std::vector<float>>& mat_from_vec);
  ~CPUMatrix();

  size_t m() const { return m_; }
  size_t n() const { return n_; }
  size_t size() const { return m_ * n_; }

  float& operator()(const size_t& x, const size_t& y);
  const float& operator()(const size_t& x, const size_t& y) const;

  CPUMatrix& operator=(const CPUMatrix& matB);
  CPUMatrix& operator+=(const CPUMatrix& matB);
  CPUMatrix& operator-=(const CPUMatrix& matB);
  CPUMatrix& operator*=(const float& scale);
  CPUMatrix operator+(const CPUMatrix& matB) const;
  CPUMatrix operator-(const CPUMatrix& matB) const;
  CPUMatrix operator*(const float& scale) const;
  CPUMatrix copy() const { return (*this); }

  CPUMatrix& reshape(const size_t& n);
  CPUMatrix& reshape(const size_t& m, const size_t& n);
  CPUMatrix& reshape(const MatrixSize& size);
  CPUMatrix& resize(const size_t& n);
  CPUMatrix& resize(const size_t& m, const size_t& n);
  CPUMatrix& resize(const MatrixSize& sz);

  CPUMatrix& ones();
  CPUMatrix& ones(const size_t& n);
  CPUMatrix& ones(const size_t& m, const size_t& n);
  CPUMatrix& ones(const MatrixSize& sz);
  CPUMatrix& zeros();
  CPUMatrix& zeros(const size_t& n);
  CPUMatrix& zeros(const size_t& m, const size_t& n);
  CPUMatrix& zeros(const MatrixSize& sz);

  CPUMatrix& arange(const float& end);
  CPUMatrix& arange(const float& start, const float& end);
  CPUMatrix& arange(const float& start, const float& end, const float& step);
  CPUMatrix& rand(const float& lower_limit, const float& upper_limit);

  CPUMatrix& fill(const float& val);
  CPUMatrix& I(const size_t& sz);
  CPUMatrix& T();
  CPUMatrix& flip(const size_t& axis);  // 0 for column, 1 for row
  CPUMatrix& rotate90(const size_t& k);
  CPUMatrix& identity(const size_t& sz) { return (*this).I(sz); }
  CPUMatrix& eye(const size_t& n) { return (*this).I(n); }
  CPUMatrix& transpose() { return (*this).T(); }
  CPUMatrix& abs();                            // Absolute
  CPUMatrix& fabs() { return (*this).abs(); }  // Floating point absolute
  // Only works with square matrix, matrix self dot
  CPUMatrix& power(const float& power);
  float sum();

  CPUMatrix col(const size_t& col_num);
  CPUMatrix cols(const size_t& col_start, const size_t& col_end);
  CPUMatrix row(const size_t& row_num);
  CPUMatrix rows(const size_t& row_start, const size_t& row_end);
  CPUMatrix submatrix(const size_t& row_start, const size_t& row_end,
                      const size_t& col_start, const size_t& col_end);

  CPUMatrix& concatenate(const CPUMatrix& matB, const size_t& axis);

  CPUMatrix& dot(const CPUMatrix& matB);
  CPUMatrix& convolution(const CPUMatrix& mask);
  CPUMatrix& convolution(const CPUMatrix& mask, const size_t& stride);
  CPUMatrix& convolution(const CPUMatrix& mask, const size_t& stride,
                         const PaddingType& padding_type);

  CPUMatrix& maxPooling(const size_t& size);
};

std::ostream& operator<<(std::ostream&, const CPUMatrix&);

#endif