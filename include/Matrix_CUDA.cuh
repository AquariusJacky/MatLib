#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include <stdlib.h>

#include <iostream>

#include "Matrix.h"

// cuda_matrix.cuh
namespace CUDA {

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
  Matrix(const ::Matrix& cpu_mat);
  Matrix(const Matrix& matB);
  ~Matrix();

  // Copy back to CPU
  void toCPU(::Matrix& cpu_mat);

  Matrix& operator=(const Matrix& matB);

  // CUDA Operations
  void fill(const float& val);
  Matrix& add(const Matrix& matB);
  Matrix dot(const Matrix& matB);
  void scale(float scalar);
  void convolution(const Matrix& mask, Matrix& result);
};

}  // namespace CUDA

#endif