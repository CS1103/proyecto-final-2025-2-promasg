cmake_minimum_required(VERSION 3.18)
project(pong_ai CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files for the neural network library
set(NN_SOURCES
    src/utec/apps/PatternClassifier.cpp
    src/utec/apps/SequencePredictor.cpp
    src/utec/apps/ControllerDemo.cpp
)

# Create the neural network library
add_library(pong_ai_lib ${NN_SOURCES})
target_include_directories(pong_ai_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Test executables
enable_testing()

# Test: Tensor
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor pong_ai_lib)
add_test(NAME TensorTests COMMAND test_tensor)

# Test: Neural Network
add_executable(test_neural_network tests/test_neural_network.cpp)
target_link_libraries(test_neural_network pong_ai_lib)
add_test(NAME NeuralNetworkTests COMMAND test_neural_network)

# Test: Applications
add_executable(test_applications tests/test_applications.cpp)
target_link_libraries(test_applications pong_ai_lib)
add_test(NAME ApplicationTests COMMAND test_applications)

# Main demo executable
add_executable(pong_ai_demo src/main.cpp)
target_link_libraries(pong_ai_demo pong_ai_lib)

# Example usage executable
add_executable(example_usage examples/example_usage.cpp)
target_link_libraries(example_usage pong_ai_lib)

# Print build information
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
