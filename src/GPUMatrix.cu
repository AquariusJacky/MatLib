#include <math.h>

#include <iostream>

#include "MatLib/GPUMatrix.cuh"

#define BLOCK_SIZE 32
#define TILE_SIZE 32

namespace GPU {

__global__ void vectorAdd(const float* A, const float* B, float* C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int testCUDA() {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  // float* h_A = (float*)malloc(size);
  float* h_A = new float[size];

  // Allocate the host input vector B
  // float* h_B = (float*)malloc(size);
  float* h_B = new float[size];

  // Allocate the host output vector C
  // float* h_C = (float*)malloc(size);
  float* h_C = new float[size];

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float* d_A = NULL;
  err = cudaMalloc((void**)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float* d_B = NULL;
  err = cudaMalloc((void**)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float* d_C = NULL;
  err = cudaMalloc((void**)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}

Matrix::Matrix(const CPU::Matrix& cpu_mat) : m_(cpu_mat.m_), n_(cpu_mat.n_) {
  allocateDeviceMemory();

  // Copy data from CPU to GPU
  cudaError_t status = cudaMemcpy(
      d_data, cpu_mat.data_, m_ * n_ * sizeof(float), cudaMemcpyHostToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data to GPU");
  }
}

Matrix::Matrix(const size_t m, const size_t n) : m_(m), n_(n) {
  allocateDeviceMemory();
}

Matrix::Matrix(const Matrix& matB) : m_(matB.m_), n_(matB.n_) {
  allocateDeviceMemory();

  // Copy data from GPU to GPU
  cudaError_t status = cudaMemcpy(d_data, matB.d_data, m_ * n_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data to GPU");
  }
}

Matrix::~Matrix() {
  m_ = 0;
  n_ = 0;
  freeDeviceMemory();
}

void Matrix::allocateDeviceMemory() {
  if (m_ * n_ > 0) {
    cudaError_t status = cudaMalloc(&d_data, m_ * n_ * sizeof(float));

    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("Failed to allocate GPU memory");
    }
  }
}

void Matrix::freeDeviceMemory() {
  if (d_data) {
    cudaFree(d_data);
    d_data = nullptr;
  }
}

void Matrix::toCPU(CPU::Matrix& cpu_mat) {
  cpu_mat = CPU::Matrix(m_, n_);

  cudaError_t status = cudaMemcpy(
      cpu_mat.data_, d_data, m_ * n_ * sizeof(float), cudaMemcpyDeviceToHost);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Failed to copy data back to CPU");
  }
}

Matrix& Matrix::operator=(const Matrix& matB) {
  freeDeviceMemory();

  m_ = matB.m_;
  n_ = matB.n_;

  allocateDeviceMemory();

  // Copy data from CPU to GPU
  cudaError_t status = cudaMemcpy(d_data, matB.d_data, m_ * n_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice);

  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    freeDeviceMemory();
    throw std::runtime_error("Failed to copy data to GPU");
  }

  return (*this);
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
    throw std::runtime_error("Kernel launch failed");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed");
  }

  return (*this);
}

// Example CUDA kernel for matrix multiplication
__global__ void matrixFillKernel(float* A, const float val, size_t m,
                                 size_t n) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    A[row * n + col] = val;
  }
}

Matrix& Matrix::fill(const float& val) {
  size_t m = m_, n = n_;

  // Set up grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  matrixFillKernel<<<dimGrid, dimBlock>>>(d_data, val, m, n);

  // Check for errors
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Kernel launch failed");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed");
  }

  return (*this);
}

// Example CUDA kernel for matrix multiplication
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
    throw std::runtime_error("Size of A does not match size of B");
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
    throw std::runtime_error("Kernel launch failed");
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
    throw std::runtime_error("col # of A doesn't match row # of B");
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
    throw std::runtime_error("Kernel launch failed");
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
      // Compare and keep the maximum
      // If the next element is larger, replace the current one
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
    throw std::runtime_error("Unsupported reduction operation");
  }

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Reduce Kernel launch failed");
  }

  // Synchronize
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Device synchronization failed");
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
      throw std::runtime_error("Unsupported reduction operation");
    }

    status = cudaGetLastError();
    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("Reduce Kernel launch failed");
    }

    // Synchronize
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("Device synchronization failed");
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
    throw std::runtime_error("Mask is not a square matrix");
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
    throw std::runtime_error("Kernel launch failed");
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
    throw std::runtime_error("Kernel launch failed");
  }

  return (*this) = result;
}

}  // namespace GPU