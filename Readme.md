# MatLib Library

A high-performance GPU matrix computation library using CUDA.

## Installation

See [Installation Guide](docs/installation.md) for detailed instructions.

Quick install:
```bash
./configure --prefix=/usr/local
make
sudo make install
```

## Usage

```cpp
#include <matlib/Matrix.h>
#include <matlib/Matrix_CUDA.cuh>

// Create and initialize CPU matrix
CPU::Matrix cpuA(1000, 1000);
cpuA.fill(7.0f);

// Transfer to GPU and compute
GPU::Matrix gpuA(cpuA);
gpuA.scale(2.0f);
```

## To Do list
- Reduction
- Scan
- histogram
- Sparse Matrix (SpMV, CSR, etc.)

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/getting-started.md)
- [Examples](examples/)
- [Performance Tests](examples/1_tester/)

## Requirements

- CUDA Toolkit (11.0 or higher)
- C++11 compatible compiler
- NVIDIA GPU (compute capability 3.0+)

## License

[MIT License](LICENSE)