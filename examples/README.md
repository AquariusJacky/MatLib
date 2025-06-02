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
make              # Build all examples
```

## Running Examples

```bash
make run-0_basic      # Run basic operations example
make run-1_tester     # Run matrix multiplication example
```

## Example Programs

1. `0_basic`: Demonstrates basic matrix operations and usage
2. `1_tester`: Performance comparison with CPU operations

## Creating Your Own Programs

To compile your own program using GPUMatrix:

```bash
g++ -std=c++11 your_program.cpp \
    -I/path/to/gpumatrix/include \
    -L/path/to/gpumatrix/lib \
    -lgpumatrix -lcudart
```

Make sure to have CUDA toolkit installed and the GPUMatrix library built.