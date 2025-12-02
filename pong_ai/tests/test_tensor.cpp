/**
 * @file test_tensor.cpp
 * @brief Casos de prueba para la clase Tensor
 */

#include "../include/utec/algebra/Tensor.h"
#include <cassert>
#include <iostream>

using namespace utec::algebra;

void test_tensor_creation() {
    std::cout << "Testing Tensor creation..." << std::endl;
    
    // Test 1D tensor
    Tensor<float, 1> t1(5);
    assert(t1.shape()[0] == 5);
    assert(t1.size() == 5);
    
    // Test 2D tensor
    Tensor<float, 2> t2(3, 4);
    assert(t2.shape()[0] == 3);
    assert(t2.shape()[1] == 4);
    assert(t2.size() == 12);
    
    // Test 3D tensor
    Tensor<float, 3> t3(2, 3, 4);
    assert(t3.shape()[0] == 2);
    assert(t3.shape()[1] == 3);
    assert(t3.shape()[2] == 4);
    assert(t3.size() == 24);
    
    std::cout << "✓ Tensor creation tests passed" << std::endl;
}

void test_tensor_access() {
    std::cout << "Testing Tensor access..." << std::endl;
    
    Tensor<float, 2> t(3, 3);
    
    // Test variadic access
    t(0, 0) = 1.0f;
    t(0, 1) = 2.0f;
    t(1, 1) = 5.0f;
    t(2, 2) = 9.0f;
    
    assert(t(0, 0) == 1.0f);
    assert(t(0, 1) == 2.0f);
    assert(t(1, 1) == 5.0f);
    assert(t(2, 2) == 9.0f);
    
    std::cout << "✓ Tensor access tests passed" << std::endl;
}

void test_tensor_operations() {
    std::cout << "Testing Tensor operations..." << std::endl;
    
    Tensor<float, 2> t1(2, 2);
    Tensor<float, 2> t2(2, 2);
    
    t1(0, 0) = 1.0f; t1(0, 1) = 2.0f;
    t1(1, 0) = 3.0f; t1(1, 1) = 4.0f;
    
    t2(0, 0) = 1.0f; t2(0, 1) = 1.0f;
    t2(1, 0) = 1.0f; t2(1, 1) = 1.0f;
    
    // Test addition
    auto t3 = t1 + t2;
    assert(t3(0, 0) == 2.0f);
    assert(t3(0, 1) == 3.0f);
    assert(t3(1, 0) == 4.0f);
    assert(t3(1, 1) == 5.0f);
    
    // Test scalar multiplication
    auto t4 = t3 * 2.0f;
    assert(t4(0, 0) == 4.0f);
    assert(t4(0, 1) == 6.0f);
    
    std::cout << "✓ Tensor operations tests passed" << std::endl;
}

void test_tensor_fill() {
    std::cout << "Testing Tensor fill..." << std::endl;
    
    Tensor<float, 2> t(3, 3);
    t.fill(5.0f);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(t(i, j) == 5.0f);
        }
    }
    
    std::cout << "✓ Tensor fill tests passed" << std::endl;
}

int main() {
    std::cout << "=== Running Tensor Tests ===" << std::endl;
    
    try {
        test_tensor_creation();
        test_tensor_access();
        test_tensor_operations();
        test_tensor_fill();
        
        std::cout << "\n=== All Tensor tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
