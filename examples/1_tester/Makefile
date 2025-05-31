# Compiler settings
CXX := g++
NVCC := nvcc

# Project paths
PROJECT_ROOT := $(realpath ../..)
CUDA_PATH := /usr/local/cuda
MATRIX_LIB_PATH := $(PROJECT_ROOT)/build/release/lib
MATRIX_INCLUDE_PATH := $(PROJECT_ROOT)/include

# Compiler flags
CXX_FLAGS := -O3 -std=c++11 -Wall
NVCC_FLAGS := -O3 -std=c++11

# Include and library paths
INCLUDES := -I$(MATRIX_INCLUDE_PATH) -I$(CUDA_PATH)/include -I.
LIB_PATHS := -L$(MATRIX_LIB_PATH) -L$(CUDA_PATH)/lib64
LIBS := -lgpumatrix -lcudart

# Source and build settings
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:%.cpp=build/%.o)
BUILD_DIR := build
TARGET := $(BUILD_DIR)/1_tester

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile cpp files
build/%.o: %.cpp | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Link target
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(OBJECTS) -o $@ $(LIB_PATHS) $(LIBS)

# Run the tester example
run: $(TARGET)
	@echo "Running 1_tester..."
	@LD_LIBRARY_PATH=$(MATRIX_LIB_PATH):$(CUDA_PATH)/lib64 ./$(TARGET)

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)

# Debug info
debug:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Include paths: $(INCLUDES)"
	@echo "Library paths: $(LIB_PATHS)"

.PHONY: all run clean debug