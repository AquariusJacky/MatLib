#include <stdlib.h>

#include <iostream>
#include <typeinfo>

#include "Matrix.h"
#include "Matrix_CUDA.cuh"

#define MATRIX_SIZE 10000

int main() {
  Matrix cpu_mat;
  cpu_mat.I(MATRIX_SIZE);

  Matrix cpu_mat2(MATRIX_SIZE, MATRIX_SIZE);
  cpu_mat2.fill(200);

  Matrix cpu_mat3(MATRIX_SIZE, MATRIX_SIZE / 2);
  cpu_mat3.fill(10.1);

  Matrix cpu_res(MATRIX_SIZE, MATRIX_SIZE);

  CUDA::Matrix gpu_mat(cpu_mat);
  CUDA::Matrix gpu_mat2(cpu_mat2);
  CUDA::Matrix gpu_mat3(cpu_mat3);
  CUDA::Matrix gpu_mat_res1;
  CUDA::Matrix gpu_mat_res2;

  Matrix res = cpu_mat + cpu_mat2;
  gpu_mat.add(gpu_mat2);
  // std::cout << gpu_mat;
  gpu_mat.toCPU(cpu_res);

  res -= cpu_res;
  float error = res.abs().sum();
  std::cout << "Error = " << error << std::endl;

  res = cpu_res.dot(cpu_mat3);
  gpu_mat_res2 = gpu_mat.dot(gpu_mat3);
  gpu_mat_res2.toCPU(cpu_res);

  res -= cpu_res;
  error = res.abs().sum();
  std::cout << "Error = " << error << std::endl;

  return 0;
}