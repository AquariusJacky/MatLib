# GPUMatrix Performance Tester

This example provides a comprehensive testing suite for the GPUMatrix library, demonstrating performance benchmarks and correctness verification.

## Components

- `tester.cpp` - Main test runner
- `MatrixTester.h` - Test suite header
- `MatrixTester.cpp` - Test implementations

## Tests Included

1. Performance Tests
   - Matrix multiplication
   - Element-wise operations
   - Memory transfer speed

2. Correctness Tests
   - Result validation against CPU computations
   - Memory leak detection
   - Error handling verification

## Building and Running

```bash
make        # Build the tester
make run    # Run all tests
make clean  # Clean build files
```

## Configuration

You can modify test parameters in `MatrixTester.h`:
```cpp
static const int MATRIX_SIZE = 1000;
static const int NUM_ITERATIONS = 100;
static const float ERROR_THRESHOLD = 1e-6;
```

## Troubleshooting

1. CUDA Device Issues:
   ```bash
   # Check CUDA device status
   nvidia-smi
   ```

2. Library Path:
   ```bash
   export LD_LIBRARY_PATH=/path/to/gpumatrix/lib:$LD_LIBRARY_PATH
   ```

## See Also

- [Basic Example](../0_basic/README.md)
- [GPUMatrix Documentation](../../docs/README.md)