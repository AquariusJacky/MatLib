# Basic MatLib Example

This example demonstrates the basic usage of the MatLib library, including matrix creation, data transfer between CPU and GPU, and basic matrix operations.

## Prerequisites

- CUDA Toolkit (11.0 or higher)
- MatLib library built and installed
- C++11 compatible compiler
- NVIDIA GPU with compute capability 3.0 or higher

## Building

```bash
make        # Build the example
make run    # Run the example
make clean  # Clean build files
```

## Example Code Explanation

The example demonstrates:
1. Record time 1
2. Creating matrices on CPU and GPU
3. Record time 2
4. Transferring data between CPU and GPU
5. Record time 3
6. Performing basic matrix operations
  - Fill
  - Scale
7. Record time 4
8. Transferring data back to CPU from GPU
9. Record time 5

```cpp
#include <chrono>
// Create CPU matrix
CPU::Matrix cpuA(1000, 1000);

// Transfer to GPU and perform operations and record their time
std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
GPU::Matrix gpuA(cpuA);
std::chrono::time_point<std::chrono::high_resolution_clock> transfer_to_gpu_time = std::chrono::high_resolution_clock::now();

cpuA.fill(7.0f);
gpuA.scale(2.0f);

// Get results back to CPU
gpuA.toCPU(cpuA);

// Get time with
std::chrono::duration<float, std::milli>(transfer_to_gpu_time -
                                                        start_time)
```

## Expected Output

```
Calculation:
  Fill FILL_VALUE
  Scale SCALE_FACTOR
Results Correct

Timing:
  Transfer to GPU:      xxx.xxx ms
  Calculation:          xxx.xxx ms
  Transfer to CPU:      xxx.xxx ms
=====================================
  Total time:           xxx.xxx ms
```

## Troubleshooting

1. If the library is not found:
   ```bash
   export LD_LIBRARY_PATH=/path/to/matlib/lib:$LD_LIBRARY_PATH
   ```

2. If CUDA is not found:
   ```bash
   export CUDA_PATH=/path/to/cuda
   ```
   default path=/usr/local/cuda

## See Also

- [MatLib Documentation](../../docs/README.md)
- [Advanced Examples](../2_tester/README.md)