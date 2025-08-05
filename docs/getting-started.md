# Using MatLib Library in Your Project

## Prerequisites

- CUDA Toolkit (11.0 or higher)
- C++11 compatible compiler (g++)
- MatLib library installed
- NVIDIA GPU with compute capability 3.0 or higher

## Project Setup

### 1. **Create Project Structure**
```bash
mkdir myproject
cd myproject
touch main.cpp
```

### 2. **Compile**

Compile with the command
```bash
g++ -std=c++11 your_program.cpp -I/path/to/matlib/include -L/path/to/matlib/lib -lmatlib -lcudart
```

Or alternatively, create a Makefile with the [Makefile-template](./Makefile-template).

## Building Your Project

```bash
# Build the program
make

# Run the program
make run

# Clean build files
make clean
```

## Project Structure
```
myproject/
├── Makefile
├── main.cpp
└── build/
    └── myprogram
```

## Troubleshooting

### Library Not Found
```bash
# Add to ~/.bashrc or run in terminal
export LD_LIBRARY_PATH=/path/to/matlib/lib:$LD_LIBRARY_PATH
```

### CUDA Installation Check
```bash
# Check CUDA compiler
nvcc --version

# Check GPU status
nvidia-smi
```

### Include Files Not Found
- Verify `MATLIB_PATH` in Makefile
- Check if headers exist in `$(MATLIB_PATH)/include/matlib/`

## Next Steps

See the MatLib library documentation for:
- API Reference
- Code Examples
- Performance Tips
- Advanced Features

## Reference
- Fastor is a high performance tensor (fixed multi-dimensional array) library for modern C++.
  https://github.com/romeric/Fastor