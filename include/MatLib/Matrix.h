#ifndef MATRIX_H
#define MATRIX_H

#include "CPUMatrix.h"
#include "GPUMatrix.cuh"

namespace matlib {
    using CPUMatrix = CPU::Matrix;
    using GPUMatrix = GPU::Matrix;
}

#endif // MATRIX_H