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
1. Creating matrices on CPU and GPU
2. Transferring data between CPU and GPU
3. Performing basic matrix operations
  - Fill
  - Scale
4. Transferring data back to CPU from GPU

```cpp
// Create CPU matrix
CPU::Matrix cpuA(1000, 1000);

// Transfer to GPU and perform operations
GPU::Matrix gpuA(cpuA);
cpuA.fill(7.0f);
gpuA.scale(2.0f);

// Get results back to CPU
gpuA.toCPU(cpuA);
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
- [Advanced Examples](../1_tester/README.md)