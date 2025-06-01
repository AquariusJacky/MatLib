# Installation Guide for GPUMatrix Library

## Prerequisites

- CUDA Toolkit (11.0 or higher)
- C++11 compatible compiler (g++)
- CMake 3.10 or higher (optional)
- NVIDIA GPU with compute capability 3.0 or higher

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/AquariusJacky/GPUMatrix.git
cd GPUMatrix

# Configure installation
./configure

# Build and install
make
sudo make install
```

### Method 2: Custom Installation Location

```bash
# Clone and enter directory
git clone https://github.com/AquariusJacky/GPUMatrix.git
cd GPUMatrix

# Configure with custom path
./configure --prefix=$HOME/.local

# Build and install
make
make install

# Add to your ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Method 3: Build Without Installing

```bash
# Clone and enter directory
git clone https://github.com/AquariusJacky/GPUMatrix.git
cd GPUMatrix

# Just build the library
make

# Library will be in build/release/lib
# Headers will be in include/gpumatrix/
```

## Configuration Options

```bash
./configure --help                          # Show all options
./configure --prefix=/path/to/install       # Custom installation path
./configure --with-cuda=/usr/local/cuda     # Custom CUDA installation
```

## Verifying Installation

```bash
# Check if library is installed
ls /usr/local/lib/libgpumatrix.so     # If installed system-wide
ls $HOME/.local/lib/libgpumatrix.so   # If installed locally

# Check if headers are installed
ls /usr/local/include/gpumatrix/*.h    # If installed system-wide
ls $HOME/.local/include/gpumatrix/*.h  # If installed locally

# Verify dynamic library
ldd /usr/local/lib/libgpumatrix.so
```

## Troubleshooting

### Common Issues

1. CUDA Not Found
```bash
# Set CUDA path explicitly
export CUDA_PATH=/usr/local/cuda
./configure --with-cuda=$CUDA_PATH
```

2. Library Not Found at Runtime
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  # For system installation
# or
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH  # For local installation
```

3. Compiler Issues
```bash
# Make sure CUDA compiler is in PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Uninstalling

```bash
# From build directory
sudo make uninstall  # For system installation
# or
make uninstall      # For local installation
```

## Next Steps

- Check out the Basic Examples
- Read the API Documentation
- Try the Performance Tests

## Additional Notes

- The library installs both static (`libgpumatrix.a`) and shared (`libgpumatrix.so`) versions
- Debug builds can be enabled with `make BUILD_TYPE=Debug`
- Installation requires write permissions to the target directory

## See Also

- [Installation Guide](docs/installation.md)
- [Examples](examples/)