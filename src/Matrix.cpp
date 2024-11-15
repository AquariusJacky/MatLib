#include "Matrix.h"

#include <cmath>
#include <iostream>

size_t Matrix::calculate_index(const size_t& x, const size_t& y) const {
  return x * n_ + y;
}

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_ = nullptr;
}

Matrix::Matrix(const size_t& m) {
  m_ = m;
  n_ = 1;
  data_ = new float[m];
  for (size_t i = 0; i < m; i++) data_[i] = 0.0f;
}

Matrix::Matrix(const size_t& m, const size_t& n) {
  m_ = m;
  n_ = n;
  data_ = new float[m * n];
  for (size_t i = 0; i < m * n; i++) data_[i] = 0.0f;
}

Matrix::Matrix(const Matrix& matB) {
  m_ = matB.m();
  n_ = matB.n();
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) = matB(i, j);
    }
  }
}

Matrix::~Matrix() { delete[] data_; }

float& Matrix::operator()(const size_t& x, const size_t& y) {
  return data_[calculate_index(x, y)];
}

const float& Matrix::operator()(const size_t& x, const size_t& y) const {
  return data_[calculate_index(x, y)];
}

Matrix& Matrix::operator=(const Matrix& matB) {
  delete[] data_;
  m_ = matB.m_;
  n_ = matB.n_;
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_ * n_; i++) data_[i] = matB.data_[i];
  return (*this);
}

Matrix& Matrix::operator+=(const Matrix& matB) {
  if (m_ != matB.m_ || n_ != matB.n_) {
    return (*this);  // Or implement error handling as needed
  }

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) += matB(i, j);
    }
  }

  return (*this);
}

Matrix& Matrix::operator-=(const Matrix& matB) {
  if (m_ != matB.m_ || n_ != matB.n_) {
    return (*this);  // Or implement error handling as needed
  }

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) -= matB(i, j);
    }
  }

  return (*this);
}

Matrix& Matrix::operator*=(const float& scale) {
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) *= scale;
    }
  }

  return (*this);
}

Matrix Matrix::operator+(const Matrix& matB) const {
  Matrix result(*this);
  result += matB;
  return result;
}

Matrix Matrix::operator-(const Matrix& matB) const {
  Matrix result(*this);
  result -= matB;
  return result;
}

Matrix Matrix::operator*(const float& scale) const {
  Matrix result(*this);
  result *= scale;
  return result;
}

Matrix& Matrix::copy(const Matrix& matB) {
  delete[] data_;
  m_ = n_ = 0;

  if (matB.size() <= 0) return (*this);

  data_ = new float[matB.size()];
  m_ = matB.m();
  n_ = matB.n();

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) = matB(i, j);
    }
  }

  return (*this);
}

Matrix& Matrix::fill(const float& val) {
  if (m_ == 0 || n_ == 0) {
    // Error handler
    return (*this);
  }

  for (size_t i = 0; i < m_ * n_; i++) {
    data_[i] = val;
  }

  return (*this);
}

Matrix& Matrix::I(const size_t& m) {
  Matrix Imat(m, m);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      Imat(i, j) = ((i == j) ? 1.0f : 0.0f);
    }
  }

  return (*this) = Imat;
}

Matrix& Matrix::T() {
  Matrix result(*this);

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) = result(i, j);
    }
  }

  return (*this);
}

Matrix& Matrix::ones() {
  (*this).fill(1);
  return (*this);
}

Matrix& Matrix::ones(const size_t& m) {
  Matrix mat(m, 1);
  mat.fill(1);
  return (*this) = mat;
}

Matrix& Matrix::ones(const size_t& m, const size_t& n) {
  Matrix mat(m, n);
  mat.fill(1);
  return (*this) = mat;
}

Matrix& Matrix::zeros() {
  (*this).fill(0);
  return (*this);
}

Matrix& Matrix::zeros(const size_t& m) {
  Matrix mat(m, 1);
  mat.fill(0);
  return (*this) = mat;
}

Matrix& Matrix::zeros(const size_t& m, const size_t& n) {
  Matrix mat(m, n);
  mat.fill(0);
  return (*this) = mat;
}

Matrix& Matrix::abs() {
  for (size_t i = 0; i < m_ * n_; i++)
    data_[i] = data_[i] >= 0 ? data_[i] : data_[i] * -1;

  return (*this);
}

float Matrix::sum() {
  float sum = 0;
  for (size_t i = 0; i < m_ * n_; i++) sum += data_[i];

  return sum;
}

Matrix& Matrix::dot(const Matrix& matB) {
  // Create error handlers
  if (!data_) return (*this);
  if (!matB.data_) return (*this);
  if (n_ != matB.m_) {
    return *this;
  }

  Matrix result(m_, matB.n_);

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < matB.n_; j++) {
      float sum = 0;
      for (size_t k = 0; k < matB.m_; k++) {
        sum += (*this)(i, k) * matB(k, j);
      }
      result(i, j) = sum;
    }
  }

  return (*this) = result;
}

// Only supports square convolution and no padding on all sides
Matrix& Matrix::convolution(const Matrix& mask) {
  if (mask.m() != mask.n() || m_ != n_ || !(m_ > mask.m())) {
    // Create error handler
    return (*this);
  }
  size_t newM = m_ + 1 - mask.m();

  Matrix result(newM, newM);

  for (size_t i = 0; i < newM; i++) {
    for (size_t j = 0; j < newM; j++) {
      float sum = 0;
      for (size_t ii = 0; ii < mask.m(); ii++) {
        for (size_t jj = 0; jj < mask.m(); jj++) {
          sum += (*this)(i + ii, j + jj) * mask(ii, jj);
        }
      }
      result(i, j) = sum;
    }
  }

  return (*this) = result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
  for (size_t i = 0; i < mat.m() - 1; i++) {
    os << (i == 0 ? "[[" : " [");
    for (size_t j = 0; j < mat.n() - 1; j++) {
      os << mat(i, j) << ", ";
    }
    os << mat(i, mat.n() - 1) << "]\n";
  }

  os << " [";
  for (size_t j = 0; j < mat.n() - 1; j++) {
    os << mat(mat.m() - 1, j) << ", ";
  }
  os << mat(mat.m() - 1, mat.n() - 1) << "]";
  os << "]\n";

  return os;
}