# MatLib Performance Tester

This example provides a comprehensive testing suite for the MatLib library, demonstrating performance benchmarks and correctness verification.

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
  MatrixTester tester;
  CPU::Matrix cpumatA(3, 3);

  // To create test
  // - createTest({Any test name you like}, {Command}, {Matrix to operate}, {useCUDA})
  tester.createTest("Matrix A Fill CUDA", "ones", cpumatA, true);
  tester.runTest("Matrix A Fill CUDA");
  tester.printResult("Matrix A Fill CUDA"); // prints the entire Matrix
  tester.printTime("Matrix A Fill CUDA");   // miliseconds
  tester.printError("Matrix A Fill CUDA");  // MSE
```

The comments are still in progress.
There are multiple commands to choose from:

"zeros", "ones", "transpose", "absolute", "sum",
"fill", "identity", "scale","maxPooling",
"addition", "dot", "convolution"

Currently "transpose", "absolute", and "sum" doesn't support GPU calculation.

CPU calculations are for time comparison.
In the future if there are more implementations of a function, it might also get a tester update for more comparison.

## Troubleshooting

1. CUDA Device Issues:
   ```bash
   # Check CUDA device status
   nvidia-smi
   ```

2. Library Path:
   ```bash
   export LD_LIBRARY_PATH=/path/to/matlib/lib:$LD_LIBRARY_PATH
   ```

## See Also

- [MatLib Documentation](../../docs/README.md)
- [Basic Example](../0_basic/README.md)