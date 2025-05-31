#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include <stdlib.h>

#include <iostream>

#include "CPUMatrix.h"

class GPUMatrix {
 private:
  float* d_data;  // Device data
  size_t m_;
  size_t n_;

  void allocateDeviceMemory();
  void freeDeviceMemory();

 public:
  GPUMatrix() : d_data(nullptr), m_(0), n_(0) {}
  GPUMatrix(const size_t m, const size_t n);
  GPUMatrix(const CPUMatrix& cpu_mat);
  GPUMatrix(const GPUMatrix& matB);
  ~GPUMatrix();

  // Copy back to CPU
  void toCPU(CPUMatrix& cpu_mat);

  GPUMatrix& operator=(const GPUMatrix& matB);

  // CUDA Operations
  void fill(const float& val);
  GPUMatrix& add(const GPUMatrix& matB);
  GPUMatrix& dot(const GPUMatrix& matB);
  GPUMatrix& scale(const float& scalar);
  GPUMatrix& convolution(const GPUMatrix& mask);
  GPUMatrix& maxPooling(const size_t& size);
};

int testCUDA();

#endif