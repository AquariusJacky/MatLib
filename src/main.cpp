#include <stdlib.h>

#include <iostream>
#include <typeinfo>

#include "Matrix.h"

int main() {
  Matrix mat;
  mat.I(3);
  mat(1, 1) = 2;

  Matrix mat2(3, 6);
  std::cout << mat2;

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      mat2(j, i) = j * i * 100;
    }
  }

  std::cout << mat;
  std::cout << mat2;
  std::cout << mat.dot(mat2);

  return 0;
}