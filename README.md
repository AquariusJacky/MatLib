# CUDAMatrix
A library that performs Matrix arithmetic that performs CUDA optimization. Provides a tester (MatrixTester) for easy time testing. Time testing uses std::chrono.

The optimization goes on...

# Run
```
mkdir build
cmake ..
make
./bin/CUDAMatrix
```

CUDA doesn't support gcc13 at the time being (11/2/2024), so I had to do this:
```
export HOST_COMPILER=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
```
