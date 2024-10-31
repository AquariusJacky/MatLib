#include "Matrix.h"

#include <iostream>

size_t Matrix::calculate_index(const size_t& x, const size_t& y) const {
  return y * n_ + x;
}

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_ = NULL;
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
  for (size_t i = 0; i < m * n; i++) data_[i] = 1.0f;
}

Matrix::Matrix(const Matrix& mat2) {
  m_ = mat2.m();
  n_ = mat2.n();
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) = mat2(j, i);
    }
  }
}

Matrix::~Matrix() { delete[] data_; }

float& Matrix::operator()(const size_t& x, const size_t& y) {
  return data_[calculate_index(x, y)];
}

const float Matrix::operator()(const size_t& x, const size_t& y) const {
  return data_[calculate_index(x, y)];
}

Matrix& Matrix::operator=(const Matrix& mat2) {
  delete[] data_;
  m_ = n_ = 0;

  if (mat2.size() <= 0) return (*this);

  data_ = new float[mat2.size()];
  m_ = mat2.m();
  n_ = mat2.n();

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) = mat2(j, i);
    }
  }

  return (*this);
}

Matrix& Matrix::operator+=(const Matrix& mat2) {
  if (m_ != mat2.m_ || n_ != mat2.n_) {
    return (*this);  // Or implement error handling as needed
  }

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) += mat2(j, i);
    }
  }

  return (*this);
}

Matrix& Matrix::operator-=(const Matrix& mat2) {
  if (m_ != mat2.m_ || n_ != mat2.n_) {
    return (*this);  // Or implement error handling as needed
  }

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) -= mat2(j, i);
    }
  }

  return (*this);
}

Matrix& Matrix::operator*=(const float& scale) {
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(j, i) *= scale;
    }
  }

  return (*this);
}

Matrix Matrix::operator+(const Matrix& mat2) const {
  Matrix result(*this);
  result += mat2;
  return result;
}

Matrix Matrix::operator-(const Matrix& mat2) const {
  Matrix result(*this);
  result += mat2;
  return result;
}

Matrix Matrix::operator*(const float& scale) const {
  Matrix result(*this);
  result *= scale;
  return result;
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
      Imat(j, i) = ((i == j) ? 1.0f : 0.0f);
    }
  }

  return (*this) = Imat;
}

Matrix& Matrix::T() {
  Matrix result(*this);

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) = result(j, i);
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

Matrix& Matrix::dot(const Matrix& mat2) {
  // Create error handlers
  if (!data_) return (*this);
  if (!mat2.data_) return (*this);
  if (n_ != mat2.m_) {
    return *this;
  }

  Matrix result(m_, mat2.n_);

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < mat2.n_; j++) {
      float sum = 0;
      for (size_t k = 0; k < mat2.m_; k++) {
        sum += (*this)(k, i) * mat2(j, k);
      }
      result(j, i) = sum;
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
          sum += (*this)(j + jj, i + ii) * mask(jj, ii);
        }
      }
      result(j, i) = sum;
    }
  }

  return (*this) = result;
}

// outside Matrix class
std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
  for (size_t i = 0; i < mat.m() - 1; i++) {
    os << (i == 0 ? "[[" : " [");
    for (size_t j = 0; j < mat.n() - 1; j++) {
      os << mat(j, i) << ", ";
    }
    os << mat(mat.n() - 1, i) << "]\n";
  }

  os << " [";
  for (size_t j = 0; j < mat.n() - 1; j++) {
    os << mat(j, mat.m() - 1) << ", ";
  }
  os << mat(mat.n() - 1, mat.m() - 1) << "]";
  os << "]\n";

  return os;
}