# Include generated configuration
-include config.mk

# Compiler settings
CXX := g++
NVCC := nvcc
AR := ar

# Compiler flags - add position independent code flag
CXX_FLAGS := -O3 -std=c++14 -fPIC
NVCC_FLAGS := -O3 -std=c++14 --compiler-options -fPIC
DEBUG_FLAGS := -g -G

# Include paths
INCLUDES := -I./include $(CUDA_INCLUDE)
TEST_INCLUDES := $(INCLUDES) -I/usr/include/gtest

# Library paths
LIB_PATHS := $(CUDA_LIBDIR)

# Build type (Release by default)
BUILD_TYPE ?= Release
ifeq ($(BUILD_TYPE), Debug)
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
CPP_SOURCES := $(wildcard src/*.cpp)
CUDA_SOURCES := $(wildcard src/*.cu)
TEST_CPP_SOURCES := $(wildcard tests/*.cpp)
TEST_CUDA_SOURCES := $(wildcard tests/*.cu)		# Add this line

# Output files
CPP_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SOURCES:.cpp=.o)))
CUDA_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_SOURCES:.cu=.cu.o)))
TEST_CPP_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(TEST_CPP_SOURCES:.cpp=.o)))
TEST_CUDA_OBJECTS := $(addprefix $(OBJ_DIR)/, $(notdir $(TEST_CUDA_SOURCES:.cu=.cu.o)))		# Add this line

TEST_OBJECTS := $(TEST_CPP_OBJECTS) $(TEST_CUDA_OBJECTS)

# Library names
LIB_NAME := libmatlib.so
STATIC_LIB_NAME := libmatlib.a
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
$(OBJ_DIR)/%.o: src/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@
	
$(OBJ_DIR)/%.cu.o: src/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: tests/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) $(TEST_INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.cu.o: tests/%.cu
		@echo "Compiling CUDA test $<..."
		$(NVCC) $(NVCC_FLAGS) $(TEST_INCLUDES) -c $< -o $@

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
	$(NVCC) $(NVCC_FLAGS) $(TEST_OBJECTS) -o $@ -L$(LIB_DIR) -lmatlib $(LIB_PATHS) -lcudart $(TEST_LIBS)

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
	cp include/MatLib/*.h include/MatLib/*.cuh $(INSTALL_INCLUDE_PATH)/
	@echo "Updating library cache..."
	ldconfig $(INSTALL_LIB_PATH)

sanitize: $(TEST_EXECUTABLE)
		@echo "Running tests with Compute Sanitizer..."
		@LD_LIBRARY_PATH=$(LIB_DIR):$(CUDA_PATH)/lib64 compute-sanitizer --tool memcheck ./$(TEST_EXECUTABLE)

# Clean targets
clean:
	@echo "Cleaning build files..."
	@rm -rf build

# Print variables for debugging
print-%:
	@echo $* = $($*)

# Declare phony targets
.PHONY: all directories clean test install print-%