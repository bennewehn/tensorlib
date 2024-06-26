# Compiler
NVCC = nvcc
CC = gcc

# Target executable name
TARGET = a.out

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LIB_DIR = lib
INCLUDE_DIR = include

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cu)

# Object files
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(filter %.cu,$(SRCS)))
C_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(filter %.c,$(SRCS)))
OBJS = $(CU_OBJS) $(C_OBJS)

# Compiler flags
NVCCFLAGS = -std=c++17 -Xcompiler -Wall,-Wextra, -arch=sm_86
INCLUDE_FLAGS = -I$(INCLUDE_DIR)
DEBUG_FLAGS = -g -O0 -DDEBUG_MODE
RELEASE_FLAGS = -O3

# Linker flags
LINKER_FLAGS = -L$(LIB_DIR)

# Default target
all: debug

# Debug target
debug: NVCCFLAGS += $(DEBUG_FLAGS)
debug: $(BIN_DIR)/$(TARGET)

# Release target
release: NVCCFLAGS += $(RELEASE_FLAGS)
release: $(BIN_DIR)/$(TARGET)

# Link the executable
$(BIN_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) $(LINKER_FLAGS) -o $@ $^

# Compile CUDA object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -c -o $@ $<

# Compile C object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) -Wall -Wextra $(INCLUDE_FLAGS) $(DEBUG_FLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all debug release clean