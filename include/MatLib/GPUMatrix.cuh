#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include <stdlib.h>

#include <iostream>

#include "CPUMatrix.h"

namespace GPU {
class Matrix {
 private:
  float* d_data;  // Device data
  size_t m_;
  size_t n_;

  void allocateDeviceMemory();
  void freeDeviceMemory();

 public:
  Matrix() : d_data(nullptr), m_(0), n_(0) {}
  Matrix(const size_t m, const size_t n);
  Matrix(const CPU::Matrix& cpu_mat);
  Matrix(const Matrix& matB);
  ~Matrix();

  size_t m() const { return m_; }
  size_t n() const { return n_; }
  size_t size() const { return m_ * n_; }

  // Copy back to CPU
  void toCPU(CPU::Matrix& cpu_mat);

  Matrix& operator=(const Matrix& matB);

  // CUDA Operations
  Matrix& fill(const float& val);
  Matrix& add(const Matrix& matB);
  Matrix& dot(const Matrix& matB);
  Matrix& scale(const float& scalar);

  float& reduction(const std::string& op) const;
  float& max() const { return reduction("max"); }
  float& min() const { return reduction("min"); }
  float& sum() const { return reduction("sum"); }

  int equal(const Matrix& matB) const;

  Matrix& convolution(const Matrix& mask);
  Matrix& maxPooling(const size_t& size);
};

int testCUDA();
}  // namespace GPU

#endif