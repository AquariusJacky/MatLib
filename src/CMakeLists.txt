# Create the executable
add_executable(${PROJECT_NAME}
    main.cpp
    Matrix.cpp
    Matrix_CUDA.cu
    
    MatrixTester.cpp
    RunningUnit.cpp
)

# Add the include directory to the include path
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)