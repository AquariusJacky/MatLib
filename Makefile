# Include generated configuration
-include config.mk

# Compiler settings
NVCC := nvcc
CXX := g++
AR := ar

# Compiler flags - add position independent code flag
NVCC_FLAGS := -O3 -std=c++11 --compiler-options -fPIC
CXX_FLAGS := -O3 -std=c++11 -fPIC
DEBUG_FLAGS := -g -G

# Include paths
INCLUDES := -I./include $(CUDA_INCLUDE)
TEST_INCLUDES := $(INCLUDES) -I/usr/include/gtest

# Library paths
LIB_PATHS := $(CUDA_LIBDIR)

# Build type (Release by default)
BUILD_TYPE ?= Release
ifeq ($(BUILD_TYPE),Debug)
	NVCC_FLAGS += $(DEBUG_FLAGS)
	CXX_FLAGS += -g
	BUILD_DIR := build/debug
else
	BUILD_DIR := build/release
endif

# Build directories
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)/lib
TEST_DIR := $(BUILD_DIR)/tests

# Source files
CUDA_SOURCES := $(wildcard src/*.cu)
CPP_SOURCES := $(wildcard src/*.cpp)
TEST_SOURCES := $(wildcard tests/*.cpp)

# Output files
CUDA_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_SOURCES:.cu=.cu.o)))
CPP_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SOURCES:.cpp=.o)))
TEST_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(TEST_SOURCES:.cpp=.o)))

# Library names
LIB_NAME := libgpumatrix.so
STATIC_LIB_NAME := libgpumatrix.a
LIB_OUTPUT := $(LIB_DIR)/$(LIB_NAME)
STATIC_LIB_OUTPUT := $(LIB_DIR)/$(STATIC_LIB_NAME)

# Test executable
TEST_EXECUTABLE := $(TEST_DIR)/test_runner
TEST_LIBS := -lgtest -lgtest_main -lpthread

# Default target
all: directories $(LIB_OUTPUT) $(STATIC_LIB_OUTPUT)

# Create build directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(TEST_DIR)

# Compilation rules
$(OBJ_DIR)/%.cu.o: src/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: tests/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) $(TEST_INCLUDES) -c $< -o $@

# Library targets
$(LIB_OUTPUT): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	@echo "Creating shared library $@..."
	$(NVCC) --shared $(NVCC_FLAGS) $^ -o $@ $(LIB_PATHS)

$(STATIC_LIB_OUTPUT): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	@echo "Creating static library $@..."
	$(AR) rcs $@ $^

# Test targets
$(TEST_EXECUTABLE): $(TEST_OBJECTS) $(LIB_OUTPUT)
	@echo "Building tests..."
	$(CXX) $(CXX_FLAGS) $(TEST_OBJECTS) -o $@ -L$(LIB_DIR) -lgpumatrix $(TEST_LIBS) $(LIB_PATHS) -lcudart

test: $(TEST_EXECUTABLE)
	@echo "Running tests..."
	@LD_LIBRARY_PATH=$(LIB_DIR):$(CUDA_PATH)/lib64 ./$(TEST_EXECUTABLE)

# Installation
install: all
	@echo "Installing libraries and headers..."
	@mkdir -p $(INSTALL_LIB_PATH)
	@mkdir -p $(INSTALL_INCLUDE_PATH)
	cp $(LIB_OUTPUT) $(INSTALL_LIB_PATH)/
	cp $(STATIC_LIB_OUTPUT) $(INSTALL_LIB_PATH)/
	cp include/gpumatrix/*.h include/gpumatrix/*.cuh $(INSTALL_INCLUDE_PATH)/
	@echo "Updating library cache..."
	ldconfig $(INSTALL_LIB_PATH)

# Clean targets
clean:
	@echo "Cleaning build files..."
	@rm -rf build

# Print variables for debugging
print-%:
	@echo $* = $($*)

# Declare phony targets
.PHONY: all directories clean test install print-%