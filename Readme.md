# GPUMatrix Library

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
#include <gpumatrix/Matrix.h>
#include <gpumatrix/Matrix_CUDA.cuh>

// Create and initialize CPU matrix
CPUMatrix cpuA(1000, 1000);
cpuA.fill(7.0f);

// Transfer to GPU and compute
GPUMatrix gpuA(cpuA);
gpuA.scale(2.0f);
```

## Documentation

- [API Reference](docs/api.md)
- [Examples](examples/)
- [Performance Tests](examples/1_tester/)

## Requirements

- CUDA Toolkit (11.0 or higher)
- C++11 compatible compiler
- NVIDIA GPU (compute capability 3.0+)

## License

[MIT License](LICENSE)