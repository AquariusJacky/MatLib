#include "gpumatrix/CPUMatrix.h"

#include <cmath>
#include <iostream>
#include <random>

/**
 * @brief Turns x, y coordinates to the 1D index of data
 * @param x x coordinate for the desired data in a matrix
 * @param y y coordinate for the desired data in a matrix
 * @return 1D Index of data
 */
size_t CPUMatrix::calculate_index(const size_t& x, const size_t& y) const {
  return x * n_ + y;
}

CPUMatrix::CPUMatrix() {
  m_ = 0;
  n_ = 0;
  data_ = nullptr;
}

CPUMatrix::CPUMatrix(const size_t& n) {
  m_ = 1;
  n_ = n;
  data_ = new float[n_];
  for (size_t i = 0; i < n_; i++) data_[i] = 0.0f;
}

CPUMatrix::CPUMatrix(const size_t& m, const size_t& n) {
  m_ = m;
  n_ = n;
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_ * n_; i++) data_[i] = 0.0f;
}

CPUMatrix::CPUMatrix(const MatrixSize& size) {
  m_ = size.m;
  n_ = size.n;
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_ * n_; i++) data_[i] = 0.0f;
}

CPUMatrix::CPUMatrix(const CPUMatrix& matB) {
  m_ = matB.m();
  n_ = matB.n();
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) = matB(i, j);
    }
  }
}
CPUMatrix::CPUMatrix(const std::vector<float>& vec) {
  m_ = 1;
  n_ = vec.size();
  data_ = new float[n_];
  for (size_t i = 0; i < n_; i++) data_[i] = vec[i];
}

CPUMatrix::CPUMatrix(const std::vector<std::vector<float>>& mat_from_vec) {
  if (mat_from_vec.size() == 0 || mat_from_vec[0].size() == 0) {
    m_ = 0;
    n_ = 0;
    data_ = nullptr;
    return;
  }

  m_ = mat_from_vec.size();
  n_ = mat_from_vec[0].size();
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) = mat_from_vec[i][j];
    }
  }
}

CPUMatrix::~CPUMatrix() { delete[] data_; }

float& CPUMatrix::operator()(const size_t& x, const size_t& y) {
  if (x >= m_ || y >= n_) {
    throw std::runtime_error("Index out of bounds");
  }
  return data_[calculate_index(x, y)];
}

const float& CPUMatrix::operator()(const size_t& x, const size_t& y) const {
  if (x >= m_ || y >= n_) {
    throw std::runtime_error("Index out of bounds");
  }
  return data_[calculate_index(x, y)];
}

CPUMatrix& CPUMatrix::operator=(const CPUMatrix& matB) {
  delete[] data_;
  m_ = matB.m_;
  n_ = matB.n_;
  data_ = new float[m_ * n_];
  for (size_t i = 0; i < m_ * n_; i++) data_[i] = matB.data_[i];
  return (*this);
}

CPUMatrix& CPUMatrix::operator+=(const CPUMatrix& matB) {
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

CPUMatrix& CPUMatrix::operator-=(const CPUMatrix& matB) {
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

CPUMatrix& CPUMatrix::operator*=(const float& scale) {
  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      (*this)(i, j) *= scale;
    }
  }

  return (*this);
}

CPUMatrix CPUMatrix::operator+(const CPUMatrix& matB) const {
  CPUMatrix result(*this);
  result += matB;
  return result;
}

CPUMatrix CPUMatrix::operator-(const CPUMatrix& matB) const {
  CPUMatrix result(*this);
  result -= matB;
  return result;
}

CPUMatrix CPUMatrix::operator*(const float& scale) const {
  CPUMatrix result(*this);
  result *= scale;
  return result;
}

CPUMatrix& CPUMatrix::reshape(const size_t& n) { return (*this).reshape(1, n); }
CPUMatrix& CPUMatrix::reshape(const size_t& m, const size_t& n) {
  if (m * n != size()) {
    // Error handle
    throw std::runtime_error("Reshape: size doesn't match");
  }

  m_ = m;
  n_ = n;

  return (*this);
}
CPUMatrix& CPUMatrix::reshape(const MatrixSize& size) {
  return (*this).reshape(size.m, size.n);
}

CPUMatrix& CPUMatrix::resize(const size_t& n) { return (*this).resize(1, n); }
CPUMatrix& CPUMatrix::resize(const size_t& m, const size_t& n) {
  size_t size = (*this).size();
  size_t new_size = m * n;
  CPUMatrix mat(m, n);
  size_t i, new_i = 0;
  while (new_i < new_size) {
    for (i = 0; i < size && new_i < new_size; i++, new_i++) {
      mat.data_[new_i] = (*this).data_[i];
    }
  }

  return (*this) = mat;
}

CPUMatrix& CPUMatrix::ones() {
  (*this).fill(1);
  return (*this);
}

CPUMatrix& CPUMatrix::ones(const size_t& n) {
  CPUMatrix mat(1, n);
  mat.fill(1);
  return (*this) = mat;
}

CPUMatrix& CPUMatrix::ones(const size_t& m, const size_t& n) {
  CPUMatrix mat(m, n);
  mat.fill(1);
  return (*this) = mat;
}

CPUMatrix& CPUMatrix::zeros() {
  (*this).fill(0);
  return (*this);
}

CPUMatrix& CPUMatrix::zeros(const size_t& n) {
  CPUMatrix mat(1, n);
  mat.fill(0);
  return (*this) = mat;
}

CPUMatrix& CPUMatrix::zeros(const size_t& m, const size_t& n) {
  CPUMatrix mat(m, n);
  mat.fill(0);
  return (*this) = mat;
}

CPUMatrix& CPUMatrix::arange(const float& end) {
  return (*this).arange(0, end, 1);
};
CPUMatrix& CPUMatrix::arange(const float& start, const float& end) {
  return (*this).arange(start, end, 1);
};

CPUMatrix& CPUMatrix::arange(const float& start, const float& end,
                             const float& step) {
  size_t n = (size_t)ceil((end - start) / step);
  CPUMatrix mat(n);
  float val = start;
  for (size_t i = 0; i < n; i++, val += step) {
    mat.data_[i] = val;
  }
  return (*this) = mat;
}
CPUMatrix& CPUMatrix::rand(const float& lower_limit, const float& upper_limit) {
  if (m_ == 0 || n_ == 0) {
    throw std::runtime_error("CPUMatrix can't have size 0");
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(lower_limit, upper_limit);

  for (int i = 0; i < (*this).size(); i++) {
    (*this).data_[i] = distribution(generator);
  }

  return (*this);
}

CPUMatrix& CPUMatrix::fill(const float& val) {
  if (m_ == 0 || n_ == 0) {
    throw std::runtime_error("CPUMatrix can't have size 0");
  }

  for (size_t i = 0; i < m_ * n_; i++) {
    data_[i] = val;
  }

  return (*this);
}

CPUMatrix& CPUMatrix::I(const size_t& sz) {
  CPUMatrix Imat(sz, sz);

  for (size_t i = 0; i < sz; i++) {
    for (size_t j = 0; j < sz; j++) {
      Imat(i, j) = ((i == j) ? 1.0f : 0.0f);
    }
  }

  return (*this) = Imat;
}

CPUMatrix& CPUMatrix::T() {
  CPUMatrix result(n_, m_);

  for (size_t i = 0; i < m_; i++) {
    for (size_t j = 0; j < n_; j++) {
      result(j, i) = (*this)(i, j);
    }
  }

  return (*this) = result;
}

/**
 * @brief Flips the matrix along the given axis.
 * @param axis The axis along which to flip over. 0 for row (up down), 1 for
 * column (left right)
 * @return The flipped matrix.
 */
CPUMatrix& CPUMatrix::flip(const size_t& axis) {
  if (axis != 0 && axis != 1) {
    throw std::runtime_error("Axis can only be 0 (row) or 1 (col)");
  }

  if (axis == 0) {
    for (size_t i = 0; i < m_ / 2; i++) {
      for (size_t j = 0; j < n_; j++) {
        std::swap((*this)(i, j), (*this)(m_ - i - 1, j));
      }
    }
  } else {
    for (size_t i = 0; i < m_; i++) {
      for (size_t j = 0; j < n_ / 2; j++) {
        std::swap((*this)(i, j), (*this)(i, n_ - j - 1));
      }
    }
  }

  return (*this);
}

/**
 * @brief Rotates the matrix counterclockwise.
 * @param k Number of times the array is rotated by 90 degrees.
 * @return The rotated matrix.
 */
CPUMatrix& CPUMatrix::rotate90(const size_t& k) {
  switch (k % 4) {
    case 1: {
      CPUMatrix result(n_, m_);
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
          result(n_ - j - 1, i) = (*this)(i, j);
        }
      }
      (*this) = result;
    } break;
    case 2: {
      CPUMatrix result(m_, n_);
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
          result(m_ - i - 1, n_ - j - 1) = (*this)(i, j);
        }
      }
      (*this) = result;
    } break;
    case 3: {
      CPUMatrix result(n_, m_);
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
          result(j, m_ - i - 1) = (*this)(i, j);
        }
      }
      (*this) = result;
    } break;
    default:
      break;
  }

  return (*this);
}

CPUMatrix& CPUMatrix::abs() {
  for (size_t i = 0; i < m_ * n_; i++) data_[i] = std::fabs(data_[i]);

  return (*this);
}

float CPUMatrix::sum() {
  float sum = 0;
  for (size_t i = 0; i < m_ * n_; i++) sum += data_[i];
  return sum;
}

/**
 * @brief Returns a 1D vector of the desired column. Shape will be preserved.
 * @param col_num The single column to be returned.
 * @return The desired column.
 */
CPUMatrix CPUMatrix::col(const size_t& col_num) {
  return (*this).submatrix(0, m_, col_num, col_num + 1);
}

/**
 * @brief Returns a matrix of the desired columns. Shape will be preserved.
 * Size will be m * (col_end - col_start).
 * @param col_start Index of the starting column.
 * @param col_end Index after the ending column.
 * @return The matrix of desired columns.
 */
CPUMatrix CPUMatrix::cols(const size_t& col_start, const size_t& col_end) {
  return (*this).submatrix(0, m_, col_start, col_end);
}

/**
 * @brief Returns a 1D vector of the desired row. Shape will be preserved.
 * @param col_num The single row to be returned.
 * @return The desired row.
 */
CPUMatrix CPUMatrix::row(const size_t& row_num) {
  return (*this).submatrix(row_num, row_num + 1, 0, n_);
}

/**
 * @brief Returns a matrix of the desired rows. Shape will be preserved. Size
 * will be (row_end - row_start) * n.
 * @param row_start Index of the starting row.
 * @param row_end Index after the ending row.
 * @return The matrix of desired rows.
 */
CPUMatrix CPUMatrix::rows(const size_t& row_start, const size_t& row_end) {
  return (*this).submatrix(row_start, row_end, 0, n_);
}

/**
 * @brief Returns the submatrix starting from start index, ending before end
 * index. Size will be (row_end - row_start) * (col_end - col_start).
 * @param row_start Starting row index
 * @param row_end Ending row index
 * @param col_start Starting column index
 * @param col_end Ending column index
 * @return The submatrix from start index to the one before end index.
 */
CPUMatrix CPUMatrix::submatrix(const size_t& row_start, const size_t& row_end,
                               const size_t& col_start, const size_t& col_end) {
  if (row_start < 0 || row_start > m_ || row_end < 0 || row_end > m_ ||
      col_start < 0 || col_start > n_ || col_end < 0 || col_end > n_) {
    throw std::runtime_error("Submatrix: Index out of bounds");
  }

  size_t new_m = row_end - row_start;
  size_t new_n = col_end - col_start;

  if (new_m <= 0 || new_n <= 0) {
    throw std::runtime_error("End index has to be larger than start index");
  }

  CPUMatrix sub_mat(new_m, new_n);
  for (size_t i = 0; i < new_m; i++) {
    for (size_t j = 0; j < new_n; j++) {
      sub_mat(i, j) = (*this)(row_start + i, col_start + j);
    }
  }

  return sub_mat;
}

/**
 * @brief Concatenates matrix B to the end of current matrix
 * @param matB CPUMatrix to be concatenated
 * @param axis The axis to concatenate. 0 for row, 1 for column
 * @return The concatenated matrix.
 */
CPUMatrix& CPUMatrix::concatenate(const CPUMatrix& matB, const size_t& axis) {
  if (axis != 0 && axis != 1) {
    throw std::runtime_error("Axis can only be 0 (row) or 1 (col)");
  }
  if ((axis == 0 && (*this).n_ != matB.n_) ||
      (axis == 1 && (*this).m_ != matB.m_)) {
    throw std::runtime_error("The size of matB doesn't math the size of matA");
  }

  size_t matASize = (*this).size();

  CPUMatrix result;
  if (axis == 0) {
    result = CPUMatrix(m_ + matB.m_, n_);
    for (size_t i = 0; i < matASize; i++) result.data_[i] = data_[i];
    for (size_t i = 0; i < matB.size(); i++)
      result.data_[matASize + i] = matB.data_[i];
  } else {
    result = CPUMatrix(m_, n_ + matB.n_);

    for (size_t i = 0; i < m_; i++) {
      for (size_t j = 0; j < n_; j++) result(i, j) = (*this)(i, j);
      for (size_t j = 0; j < matB.n_; j++) result(i, m_ + j) = matB(i, j);
    }
  }

  return (*this) = result;
}

CPUMatrix& CPUMatrix::dot(const CPUMatrix& matB) {
  // Create error handlers
  if (!data_) return (*this);
  if (!matB.data_) return (*this);
  if (n_ != matB.m_) {
    return *this;
  }

  CPUMatrix result(m_, matB.n_);

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

CPUMatrix& CPUMatrix::convolution(const CPUMatrix& mask) {
  return (*this).convolution(mask, 1, PaddingType::NONE);
}
CPUMatrix& CPUMatrix::convolution(const CPUMatrix& mask, const size_t& stride) {
  return (*this).convolution(mask, stride, PaddingType::NONE);
}

// Only supports square convolution and no padding on all sides
CPUMatrix& CPUMatrix::convolution(const CPUMatrix& mask, const size_t& stride,
                                  const PaddingType& padding_type) {
  if (m_ != n_) {
    throw std::runtime_error("Input matrix has to be square matrix");
  }
  if (mask.m() != mask.n()) {
    throw std::runtime_error("Mask has to be square matrix");
  }
  if (padding_type != FULL && !(m_ > mask.m())) {
    throw std::runtime_error(
        "Input size has to be larger or equal to mask size");
  }

  CPUMatrix result;
  size_t maskSize = mask.m();

  if (padding_type == PaddingType::NONE) {
    size_t newM = m_ + 1 - maskSize;
    result = CPUMatrix(newM, newM);

    for (size_t i = 0; i < newM; i++) {
      for (size_t j = 0; j < newM; j++) {
        float sum = 0;
        for (size_t ii = 0; ii < maskSize; ii++) {
          for (size_t jj = 0; jj < maskSize; jj++) {
            sum += (*this)(i + ii, j + jj) * mask(ii, jj);
          }
        }
        result(i, j) = sum;
      }
    }
  } else if (padding_type == PaddingType::FULL) {
    size_t newM = m_ + maskSize - 1;
    result = CPUMatrix(newM, newM);
    for (size_t i = 0; i < newM; i++) {
      for (size_t j = 0; j < newM; j++) {
        float sum = 0;
        for (size_t ii = 0; ii < maskSize; ii++) {
          for (size_t jj = 0; jj < maskSize; jj++) {
            if (i + ii < maskSize - 1 || j + jj < maskSize - 1) continue;
            if (i + ii >= (m_ + maskSize - 1) || j + jj >= (n_ + maskSize - 1))
              continue;
            sum += (*this)(i + ii + maskSize - 1, j + jj + maskSize - 1) *
                   mask(ii, jj);
          }
        }
        result(i, j) = sum;
      }
    }
  }

  return (*this) = result;
}

/**
 * @brief Concatenates matrix B to the end of current matrix
 * @param matB CPUMatrix to be concatenated
 * @param axis The axis to concatenate. 0 for row, 1 for column
 * @return The concatenated matrix.
 */
CPUMatrix& CPUMatrix::maxPooling(const size_t& pooling_size) {
  if (pooling_size > m_ || pooling_size > n_) {
    throw std::runtime_error("Pooling size can't be larger than matrix size");
  }

  size_t new_m = m_ / pooling_size, new_n = n_ / pooling_size;

  CPUMatrix result(new_m, new_n);

  for (size_t i = 0; i < new_m; i++) {
    for (size_t j = 0; j < new_n; j++) {
      float max_num = (*this)(i * pooling_size, j * pooling_size);
      for (size_t k = 0; k < pooling_size; k++) {
        for (size_t l = 0; l < pooling_size; l++) {
          float curr_num = (*this)(i * pooling_size + k, j * pooling_size + l);
          max_num = std::max(max_num, curr_num);
        }
      }
      result(i, j) = max_num;
    }
  }
  return (*this) = result;
}

std::ostream& operator<<(std::ostream& os, const CPUMatrix& mat) {
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