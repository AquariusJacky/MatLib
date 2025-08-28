#include <math.h>

#include <iostream>

#include "MatLib/GPUMatrix.cuh"

#define BLOCK_SIZE 32
#define TILE_SIZE 32

namespace GPU {

Matrix::Matrix(const CPU::Matrix& cpu_mat) : m_(cpu_mat.m_), n_(cpu_mat.n_) {
  allocateDeviceMemory();

  // Copy data from CPU to GPU
  cudaError_t status = cudaMemcpy(
      d_data, cpu_mat.data_, m_ * n_ * sizeof(float), cudaMemcpyHostToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data to GPU in CPU constructor");
  }
}

Matrix::Matrix(const size_t m, const size_t n) : m_(m), n_(n) {
  allocateDeviceMemory();
}

Matrix::Matrix(const size_t n) : m_(1), n_(n) { allocateDeviceMemory(); }

Matrix::Matrix(const Matrix& matB) : m_(matB.m_), n_(matB.n_) {
  allocateDeviceMemory();

  // Copy data from GPU to GPU
  cudaError_t status = cudaMemcpy(d_data, matB.d_data, m_ * n_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data to GPU in copy constructor");
  }
}

Matrix::~Matrix() {
  m_ = 0;
  n_ = 0;
  freeDeviceMemory();
}

void Matrix::allocateDeviceMemory() {
  freeDeviceMemory();
  if (m_ * n_ > MAX_TOTAL_ELEMENTS) {
    throw std::runtime_error(
        "Matrix size exceeds maximum allowed elements in allocateDeviceMemory");
  }
  if (m_ * n_ > 0) {
    cudaError_t status = cudaMalloc(&d_data, m_ * n_ * sizeof(float));

    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error(
          "Failed to allocate GPU memory in allocateDeviceMemory");
    }
  }
}

void Matrix::freeDeviceMemory() {
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, d_data);
  if (status != cudaSuccess) {
    std::cout << "Invalid pointer: " << cudaGetErrorString(status) << std::endl;
    cudaGetLastError();  // Clear error
  }

  if (attributes.type == cudaMemoryTypeUnregistered) return;

  status = cudaFree(d_data);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Failed to free GPU memory in freeDeviceMemory");
  }
  d_data = nullptr;
}

void Matrix::toCPU(CPU::Matrix& cpu_mat) {
  cpu_mat = CPU::Matrix(m_, n_);

  cudaError_t status = cudaMemcpy(
      cpu_mat.data_, d_data, m_ * n_ * sizeof(float), cudaMemcpyDeviceToHost);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data back to CPU in toCPU");
  }
}

Matrix& Matrix::operator=(const Matrix& matB) {
  if (this == &matB) return (*this);  // Handle self-assignment
  
  m_ = matB.m_;
  n_ = matB.n_;

  allocateDeviceMemory();

  // Copy data from CPU to GPU
  cudaError_t status = cudaMemcpy(d_data, matB.d_data, m_ * n_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error(
        "Failed to copy data to GPU in assignment operator");
  }

  return (*this);
}

// Example CUDA kernel for matrix multiplication
__global__ void matrixEqualKernel(float* A, float* B, int* result, size_t m,
                                  size_t n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    int index = row * n + col;

    // If any element is different, set result to 0 (false)
    if (A[index] != B[index]) {
      atomicExch(result, 0);
    }
  }
}

int Matrix::equal(const Matrix& matB) const {
  size_t m = m_, n = n_;

  if (matB.m_ != m || matB.n_ != n) return 0;
  if (m * n == 0) return 1;

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  int* d_result;  // Assume equal initially
  cudaMalloc(&d_result, sizeof(int));
  int initial_value = 1;
  cudaMemcpy(d_result, &initial_value, sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  matrixEqualKernel<<<dimGrid, dimBlock>>>(d_data, matB.d_data, d_result, m, n);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in equal");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed in equal");
  }

  int h_result;
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  return h_result;
}

// CUDA kernel for filling a matrix with a constant value
__global__ void matrixFillKernel(float* A, const float val, size_t m, size_t n) {
    // Calculate global thread indices
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < m && col < n) {
        size_t idx = row * n + col;  // Row-major indexing
        A[idx] = val;
            }
}

Matrix& Matrix::fill(const float& val) {
    if (m_ * n_ == 0) return *this;  // Handle empty matrix case
    if (d_data == nullptr) {
        throw std::runtime_error("Device memory not allocated in fill");
    }

    size_t m = m_, n = n_;

    // Use 2D grid configuration (corrected from your comment about 1D)
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Calculate grid dimensions - need to round up to cover all elements
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel with 2D configuration
    matrixFillKernel<<<dimGrid, dimBlock>>>(d_data, val, m, n);

    // Check for errors
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
        throw std::runtime_error("Kernel launch failed in fill");
    }

    // Synchronize
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
        throw std::runtime_error("Device synchronization failed in fill");
    }

    return *this;
}

// Example CUDA kernel for matrix multiplication
__global__ void matrixScaleKernel(float* A, const float val, size_t m,
                                  size_t n) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    A[row * n + col] = A[row * n + col] * val;
  }
}

Matrix& Matrix::scale(const float& scalar) {
  size_t m = m_, n = n_;

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  matrixScaleKernel<<<dimGrid, dimBlock>>>(d_data, scalar, m, n);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in scale");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed in scale");
  }

  return (*this);
}

__global__ void addKernel(const float* A, const float* B, float* output,
                          size_t m, size_t n) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    output[row * n + col] = A[row * n + col] + B[row * n + col];
  }
  __syncthreads();
}

Matrix& Matrix::add(const Matrix& matB) {
  size_t m = m_, n = n_;

  if (matB.m_ != m || matB.n_ != n) {
    throw std::runtime_error("Size of A does not match size of B in add");
  }

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  Matrix result(m, n);

  // Launch kernel
  addKernel<<<dimGrid, dimBlock>>>(d_data, matB.d_data, result.d_data, m, n);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in add");
  }

  return (*this) = result;
}

// Example CUDA kernel for matrix multiplication
__global__ void dotKernel(const float* A, const float* B, float* output,
                          size_t m, size_t k, size_t n) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_SIZE][TILE_SIZE];
  __shared__ float subTileB[TILE_SIZE][TILE_SIZE];

  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  size_t row = blockIdx.x * blockDim.x + tx;
  size_t col = blockIdx.y * blockDim.y + ty;

  float sum = 0;

  for (int tileIdx = 0; tileIdx < ceil((float)k / TILE_SIZE); tileIdx++) {
    // Fill in 0 if out of A's range
    if (row < m && (tileIdx * TILE_SIZE + ty) < k) {
      subTileA[tx][ty] = A[row * k + (tileIdx * TILE_SIZE + ty)];
    } else
      subTileA[tx][ty] = 0;

    // Fill in 0 if out of B's range
    if ((tileIdx * TILE_SIZE + tx) < k && col < n) {
      subTileB[tx][ty] = B[(tileIdx * TILE_SIZE + tx) * n + col];
    } else
      subTileB[tx][ty] = 0;

    __syncthreads();

    // Even calculate when out of range
    // Avoid control branch divergence
    for (int k = 0; k < TILE_SIZE; k++)
      sum += subTileA[tx][k] * subTileB[k][ty];

    __syncthreads();
  }

  if (row < m && col < n) output[row * n + col] = sum;
}

Matrix& Matrix::dot(const Matrix& matB) {
  size_t m = m_, k = n_, n = matB.n_;

  if (matB.m_ != k) {
    throw std::runtime_error("col # of A doesn't match row # of B in dot");
  }

  Matrix result(m, n);

  // Set up grid and block dimensions
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  dotKernel<<<dimGrid, dimBlock>>>(d_data, matB.d_data, result.d_data, m, k, n);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in dot");
  }

  return (*this) = result;
}

__global__ void reductionMaxKernel(float* input, float* output, int len) {
  __shared__ float partialSum[2 * BLOCK_SIZE];

  size_t t = threadIdx.x;
  size_t start = 2 * blockDim.x * blockIdx.x;

  // Load first block of elements into shared memory
  if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = -INFINITY;

  // Load second block of elements into shared memory
  if (start + t + blockDim.x < len)
    partialSum[t + blockDim.x] = input[start + t + blockDim.x];
  else
    partialSum[t + blockDim.x] = -INFINITY;

  // Sync threads to ensure all data is loaded
  __syncthreads();

  for (size_t stride = blockDim.x; stride >= 1; stride >>= 1) {
    if (t < stride) {
      // Compare and keep the maximum
      // If the next element is larger, replace the current one
      if (partialSum[t + stride] > partialSum[t])
        partialSum[t] = partialSum[t + stride];
    }
    __syncthreads();
  }

  output[blockIdx.x] = partialSum[0];
}

__global__ void reductionMinKernel(float* input, float* output, int len) {
  __shared__ float partialSum[2 * BLOCK_SIZE];

  size_t t = threadIdx.x;
  size_t start = 2 * blockDim.x * blockIdx.x;

  // Load first block of elements into shared memory
  if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = INFINITY;

  // Load second block of elements into shared memory
  if (start + t + blockDim.x < len)
    partialSum[t + blockDim.x] = input[start + t + blockDim.x];
  else
    partialSum[t + blockDim.x] = INFINITY;

  // Sync threads to ensure all data is loaded
  __syncthreads();

  for (size_t stride = blockDim.x; stride >= 1; stride >>= 1) {
    if (t < stride) {
      // Compare and keep the minimum
      // If the next element is smaller, replace the current one
      if (partialSum[t + stride] < partialSum[t])
        partialSum[t] = partialSum[t + stride];
    }
    __syncthreads();
  }

  output[blockIdx.x] = partialSum[0];
}

__global__ void reductionSumKernel(float* input, float* output, int len) {
  __shared__ float partialSum[2 * BLOCK_SIZE];

  size_t t = threadIdx.x;
  size_t start = 2 * blockDim.x * blockIdx.x;

  // Load first block of elements into shared memory
  if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = 0;

  // Load second block of elements into shared memory
  if (start + t + blockDim.x < len)
    partialSum[t + blockDim.x] = input[start + t + blockDim.x];
  else
    partialSum[t + blockDim.x] = 0;

  // Sync threads to ensure all data is loaded
  __syncthreads();

  for (size_t stride = blockDim.x; stride >= 1; stride >>= 1) {
    if (t < stride) {
      partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();
  }

  output[blockIdx.x] = partialSum[0];
}

Matrix Matrix::reduction(const std::string& op) const {

  if (m_ * n_ == 0) {
    throw std::runtime_error("Cannot reduce an empty matrix");
  }

  dim3 dimBlock(BLOCK_SIZE);
  // Number of blocks needed
  // Each block will handle 2 * BLOCK_SIZE * BLOCK_SIZE elements
  dim3 dimGrid(((m_ * n_) + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2));

  Matrix results(dimGrid.x);
  cudaError_t status;

  // Launch kernel
  if (op == "max") {
    reductionMaxKernel<<<dimGrid, dimBlock>>>(d_data, results.d_data, m_ * n_);
  } else if (op == "min") {
    reductionMinKernel<<<dimGrid, dimBlock>>>(d_data, results.d_data, m_ * n_);
  } else if (op == "sum") {
    reductionSumKernel<<<dimGrid, dimBlock>>>(d_data, results.d_data, m_ * n_);
  } else {
    throw std::runtime_error("Unsupported reduction operation in reduction");
  }

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Reduce Kernel launch failed in reduction");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed in reduction");
  }
  Matrix finalResult(1);

  // Do a second reduction if more than one block
  if (dimGrid.x > 1) {
    // Launch kernel
    if (op == "max") {
      reductionMaxKernel<<<1, dimBlock>>>(results.d_data, finalResult.d_data,
                                          dimGrid.x);
    } else if (op == "min") {
      reductionMinKernel<<<1, dimBlock>>>(results.d_data, finalResult.d_data,
                                          dimGrid.x);
    } else if (op == "sum") {
      reductionSumKernel<<<1, dimBlock>>>(results.d_data, finalResult.d_data,
                                          dimGrid.x);
    } else {
      throw std::runtime_error("Unsupported reduction operation in reduction");
    }

    status = cudaGetLastError();
    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("Reduce Kernel launch failed in second reduction");
    }

    // Synchronize
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("Device synchronization failed in reduction");
    }
  }

  if (dimGrid.x > 1)
    return finalResult;
  else
    return results;
}

// Example CUDA kernel for matrix multiplication
__global__ void convolutionKernel(const float* A, const float* mask,
                                  float* output, size_t m, size_t n, size_t k) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  size_t out_m = m - k + 1;
  size_t out_n = n - k + 1;

  if (row < out_m && col < out_n) {
    float sum = 0.0f;
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < k; j++) {
        // A[row + i][col + j] * mask[i][j]
        sum += A[(row + i) * n + (col + j)] * mask[i * k + j];
      }
    }
    output[row * out_n + col] = sum;
  }
  __syncthreads();
}

Matrix& Matrix::convolution(const Matrix& mask) {
  size_t m = m_, n = n_, k = mask.m_;

  if (mask.n_ != k) {
    throw std::runtime_error("Mask is not a square matrix in convolution");
  }

  size_t out_m = m - k + 1;
  size_t out_n = n - k + 1;

  Matrix result(out_m, out_n);

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  convolutionKernel<<<dimGrid, dimBlock>>>(d_data, mask.d_data, result.d_data,
                                           m, n, k);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in convolution");
  }

  return (*this) = result;
}

// Example CUDA kernel for matrix multiplication
__global__ void maxPoolingKernel(const float* A, float* output, size_t m,
                                 size_t n, size_t size) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  size_t out_m = m / size;
  size_t out_n = n / size;

  if (row < out_m && col < out_n) {
    float max_num = A[(row * size) * n + (col * size)];
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        float curr_num = A[((row * size) + i) * n + (col * size) + j];
        if (curr_num > max_num) max_num = curr_num;
      }
    }
    output[row * out_n + col] = max_num;
  }
  __syncthreads();
}

Matrix& Matrix::maxPooling(const size_t& size) {
  size_t m = m_, n = n_;

  size_t out_m = m_ / size;
  size_t out_n = n_ / size;

  Matrix result(out_m, out_n);

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  maxPoolingKernel<<<dimGrid, dimBlock>>>(d_data, result.d_data, m, n, size);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed in maxPooling");
  }

  return (*this) = result;
}

}  // namespace GPU