# GPUMatrix Examples

This directory contains example programs demonstrating the usage of GPUMatrix library.

## Building Examples

First, build the main library:
```bash
cd ..              # Go to root directory
make              # Build the library
make install
```

You may also change installation directories by using configure
```bash
./configure prefix=desired/path/for/installation
```
Default will be in /usr/local

Then build the examples:
```bash
cd examples       # Return to examples directory
make             # Build all examples
```

## Running Examples

```bash
./basic_operations    # Run basic operations example
./matrix_multiply    # Run matrix multiplication example
```

## Example Programs

1. `basic_operations.cpp`: Demonstrates basic matrix operations
2. `matrix_multiply.cpp`: Shows matrix multiplication usage
3. `benchmark.cpp`: Performance comparison with CPU operations

## Creating Your Own Programs

To compile your own program using GPUMatrix:

```bash
g++ -std=c++11 your_program.cpp \
    -I/path/to/gpumatrix/include \
    -L/path/to/gpumatrix/lib \
    -lgpumatrix -lcudart
```

Make sure to have CUDA toolkit installed and the GPUMatrix library built.