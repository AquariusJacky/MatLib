#include "Matrix.h"

#include <cmath>
#include <iostream>

/**
 * @brief Turns x, y coordinates to the 1D index of data
 * @param x x coordinate for the desired data in a matrix
 * @param y y coordinate for the desired data in a matrix
 * @return 1D Index of data
 */
size_t Matrix::calculate_index(const size_t& x, const size_t& y) const {
  return x * n_ + y;
}

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_ = nullptr;
}

Matrix::Matrix(const size_t& n) {
  m_ = 1;
  n_ = n;
  data_ = new float[n];
  for (size_t i = 0; i < n; i++) data_[i] = 0.0f;
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
  if (x < 0 || x >= m_ || y < 0 || y >= n_) {
    throw std::logic_error("Index out of bounds");
  }
  return data_[calculate_index(x, y)];
}

const float& Matrix::operator()(const size_t& x, const size_t& y) const {
  if (x < 0 || x >= m_ || y < 0 || y >= n_) {
    throw std::logic_error("Index out of bounds");
  }
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

Matrix& Matrix::reshape(const size_t& n) { return (*this).reshape(1, n); }
Matrix& Matrix::reshape(const size_t& m, const size_t& n) {
  if (m * n != size()) {
    // Error handle
    throw std::logic_error("Reshape: size doesn't match");
  }

  m_ = m;
  n_ = n;

  return (*this);
}
Matrix& Matrix::resize(const size_t& n) { return (*this).resize(1, n); }
Matrix& Matrix::resize(const size_t& m, const size_t& n) {
  size_t size = (*this).size();
  size_t new_size = m * n;
  Matrix mat(m, n);
  size_t i = 0, new_i = 0;
  while (new_i < new_size) {
    for (size_t i = 0; i < size && new_i < new_size; i++, new_i++) {
      mat.data_[new_i] = (*this).data_[i];
    }
  }

  return (*this) = mat;
}

Matrix& Matrix::ones() {
  (*this).fill(1);
  return (*this);
}

Matrix& Matrix::ones(const size_t& n) {
  Matrix mat(1, n);
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

Matrix& Matrix::zeros(const size_t& n) {
  Matrix mat(1, n);
  mat.fill(0);
  return (*this) = mat;
}

Matrix& Matrix::zeros(const size_t& m, const size_t& n) {
  Matrix mat(m, n);
  mat.fill(0);
  return (*this) = mat;
}

Matrix& Matrix::arange(const float& end) { return (*this).arange(0, end, 1); };
Matrix& Matrix::arange(const float& start, const float& end) {
  return (*this).arange(start, end, 1);
};

Matrix& Matrix::arange(const float& start, const float& end,
                       const float& step) {
  size_t n = (size_t)ceil((end - start) / step);
  Matrix mat(n);
  float val = start;
  for (size_t i = 0; i < n; i++, val += step) {
    mat.data_[i] = val;
  }
  return (*this) = mat;
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
  size_t m = mat.m(), n = mat.n();

  os << "[[" << mat(0, 0);
  for (size_t j = 1; j < n; j++) {
    os << ", " << mat(0, j);
  }
  if (m == 1) {
    os << "]]\n";
    return os;
  } else {
    os << "],\n";
  }
  for (size_t i = 1; i < m - 1; i++) {
    os << " [" << mat(i, 0);
    for (size_t j = 1; j < n; j++) {
      os << ", " << mat(i, j);
    }
    os << "],\n";
  }
  os << " [" << mat(m - 1, 0);
  for (size_t j = 1; j < n; j++) {
    os << ", " << mat(m - 1, j);
  }
  os << "]]\n";

  return os;
}