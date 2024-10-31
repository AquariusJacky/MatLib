#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

#include <iostream>

class Matrix {
 private:
  size_t m_;
  size_t n_;
  float* data_;  // Stores the actual data in row-major order

  size_t calculate_index(const size_t& x, const size_t& y) const;

 public:
  Matrix();
  Matrix(const size_t& m);
  Matrix(const size_t& m, const size_t& n);
  Matrix(const Matrix& mat2);
  ~Matrix();

  size_t m() const { return m_; }
  size_t n() const { return n_; }
  size_t size() const { return m_ * n_; }

  float& operator()(const size_t& x, const size_t& y);
  const float operator()(const size_t& x, const size_t& y) const;

  Matrix& operator=(const Matrix& mat2);
  Matrix& operator+=(const Matrix& mat2);
  Matrix& operator-=(const Matrix& mat2);
  Matrix& operator*=(const float& scale);
  Matrix operator+(const Matrix& mat2) const;
  Matrix operator-(const Matrix& mat2) const;
  Matrix operator*(const float& scale) const;

  Matrix& ones();
  Matrix& ones(const size_t& m);
  Matrix& ones(const size_t& m, const size_t& n);
  Matrix& zeros();
  Matrix& zeros(const size_t& m);
  Matrix& zeros(const size_t& m, const size_t& n);

  Matrix& fill(const float& val);
  Matrix& I(const size_t& m);
  Matrix& T();
  Matrix& identity(const size_t& m) { return (*this).I(m); }
  Matrix& eye(const size_t& m) { return (*this).I(m); }
  Matrix& transpose() { return (*this).T(); }

  Matrix& dot(const Matrix& mat2);
  Matrix& convolution(const Matrix& mask);
};

std::ostream& operator<<(std::ostream&, const Matrix&);

#endif