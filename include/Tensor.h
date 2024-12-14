#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

#include "Matrix.h"

class Tensor {
 public:
  Tensor(std::vector<Matrix*> matrices);
  Tensor(std::vector<Tensor*> tensors);
  ~Tensor() {}

 private:
  size_t dimension;  // 2 for matrix
  std::vector<Tensor*> data;
};

#endif