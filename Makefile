.PHONY: xor mnist test clean

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra -Wpedantic
SRC_DIR = src
BUILD_DIR = build

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile tensor.cpp to object file
$(BUILD_DIR)/tensor.o: $(SRC_DIR)/tensor.cpp $(SRC_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/tensor.cpp -o $(BUILD_DIR)/tensor.o

# Build and run XOR
xor: $(BUILD_DIR)/tensor.o $(SRC_DIR)/xor.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/xor.cpp $(BUILD_DIR)/tensor.o -o $(BUILD_DIR)/xor
	@./$(BUILD_DIR)/xor

# Build and run MNIST
mnist: $(BUILD_DIR)/tensor.o $(SRC_DIR)/mnist.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/mnist.cpp $(BUILD_DIR)/tensor.o -o $(BUILD_DIR)/mnist
	@./$(BUILD_DIR)/mnist

# Build and run tests
test: $(BUILD_DIR)/tensor.o $(SRC_DIR)/tests.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/tests.cpp $(BUILD_DIR)/tensor.o -o $(BUILD_DIR)/tests
	@./$(BUILD_DIR)/tests

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
