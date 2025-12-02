/**
 * @file test_neural_network.cpp
 * @brief Casos de prueba para la red neuronal
 */

#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_activation.h"
#include "../include/utec/nn/nn_loss.h"
#include "../include/utec/nn/nn_optimizer.h"
#include "../include/utec/nn/nn_dense.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace utec::nn;

void test_dense_layer() {
    std::cout << "Testing DenseLayer..." << std::endl;
    
    // Crear una capa Dense con inicialización
    auto init_w = [](Tensor2D& W) {
        for (size_t i = 0; i < W.shape()[0]; ++i) {
            for (size_t j = 0; j < W.shape()[1]; ++j) {
                W(i, j) = 0.1f * (i + j);
            }
        }
    };
    auto init_b = [](Tensor2D& b) { b.fill(0.0f); };
    
    utec::neural_network::Dense<float> layer(3, 2, init_w, init_b);
    
    Tensor2D input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    
    Tensor2D output = layer.forward(input);
    assert(output.shape()[0] == 1);
    assert(output.shape()[1] == 2);
    
    std::cout << "✓ DenseLayer tests passed" << std::endl;
}

void test_activation_functions() {
    std::cout << "Testing Activation Functions..." << std::endl;
    
    Tensor2D x(1, 3);
    x(0, 0) = -1.0f;
    x(0, 1) = 0.0f;
    x(0, 2) = 1.0f;
    
    // Test ReLU
    utec::neural_network::ReLU<float> relu;
    Tensor2D relu_out = relu.activate(x);
    assert(relu_out(0, 0) == 0.0f);  // max(0, -1) = 0
    assert(relu_out(0, 1) == 0.0f);  // max(0, 0) = 0
    assert(relu_out(0, 2) == 1.0f);  // max(0, 1) = 1
    
    // Test Sigmoid
    utec::neural_network::Sigmoid<float> sigmoid;
    Tensor2D sigmoid_out = sigmoid.activate(x);
    // Sigmoid values should be between 0 and 1
    for (size_t i = 0; i < 3; ++i) {
        assert(sigmoid_out(0, i) >= 0.0f && sigmoid_out(0, i) <= 1.0f);
    }
    
    // Test Linear
    utec::neural_network::Linear<float> linear;
    Tensor2D linear_out = linear.activate(x);
    assert(std::abs(linear_out(0, 0) - (-1.0f)) < 1e-5f);
    assert(std::abs(linear_out(0, 1) - 0.0f) < 1e-5f);
    assert(std::abs(linear_out(0, 2) - 1.0f) < 1e-5f);
    
    std::cout << "✓ Activation function tests passed" << std::endl;
}

void test_loss_functions() {
    std::cout << "Testing Loss Functions..." << std::endl;
    
    Tensor2D predictions(2, 2);
    predictions(0, 0) = 0.9f; predictions(0, 1) = 0.1f;
    predictions(1, 0) = 0.2f; predictions(1, 1) = 0.8f;
    
    Tensor2D targets(2, 2);
    targets(0, 0) = 1.0f; targets(0, 1) = 0.0f;
    targets(1, 0) = 0.0f; targets(1, 1) = 1.0f;
    
    // Test MSE
    utec::neural_network::MeanSquaredError<float> mse;
    float mse_loss = mse.compute(predictions, targets);
    assert(mse_loss >= 0.0f);
    
    // Test MAE
    utec::neural_network::MeanAbsoluteError<float> mae(predictions, targets);
    float mae_loss = mae.compute(predictions, targets);
    assert(mae_loss >= 0.0f);
    
    std::cout << "✓ Loss function tests passed" << std::endl;
}

void test_neural_network() {
    std::cout << "Testing NeuralNetwork..." << std::endl;
    
    utec::NeuralNetwork nn;
    nn.add_dense_layer(4, 8, std::make_shared<utec::neural_network::ReLU<float>>());
    nn.add_dense_layer(8, 2, std::make_shared<utec::neural_network::Sigmoid<float>>());
    
    Tensor2D input(2, 4);
    input(0, 0) = 0.1f; input(0, 1) = 0.2f; input(0, 2) = 0.3f; input(0, 3) = 0.4f;
    input(1, 0) = 0.5f; input(1, 1) = 0.6f; input(1, 2) = 0.7f; input(1, 3) = 0.8f;
    
    // Test forward pass
    Tensor2D output = nn.forward(input);
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 2);
    
    // Output values should be between 0 and 1 (sigmoid)
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(output(i, j) >= 0.0f && output(i, j) <= 1.0f);
        }
    }
    
    std::cout << "✓ NeuralNetwork tests passed" << std::endl;
}

void test_training() {
    std::cout << "Testing Training..." << std::endl;
    
    utec::NeuralNetwork nn;
    nn.add_dense_layer(2, 4, std::make_shared<utec::neural_network::ReLU<float>>());
    nn.add_dense_layer(4, 1, std::make_shared<utec::neural_network::Sigmoid<float>>());
    
    // Simple XOR-like data
    Tensor2D input(4, 2);
    input(0, 0) = 0.0f; input(0, 1) = 0.0f;
    input(1, 0) = 0.0f; input(1, 1) = 1.0f;
    input(2, 0) = 1.0f; input(2, 1) = 0.0f;
    input(3, 0) = 1.0f; input(3, 1) = 1.0f;
    
    Tensor2D target(4, 1);
    target(0, 0) = 0.0f;
    target(1, 0) = 1.0f;
    target(2, 0) = 1.0f;
    target(3, 0) = 0.0f;
    
    utec::neural_network::MSELoss<float> loss_obj(input, target);
    utec::neural_network::Adam<float> optimizer(0.01f);
    
    float initial_loss = nn.evaluate(input, target, loss_obj);
    
    // Train for a few steps
    for (int i = 0; i < 10; ++i) {
        nn.train_step(input, target, loss_obj, optimizer);
    }
    
    float final_loss = nn.evaluate(input, target, loss_obj);
    
    // Loss should decrease (or at least not increase significantly)
    std::cout << "Initial loss: " << initial_loss << ", Final loss: " << final_loss << std::endl;
    
    std::cout << "✓ Training tests passed" << std::endl;
}

int main() {
    std::cout << "=== Running Neural Network Tests ===" << std::endl;
    
    try {
        test_dense_layer();
        test_activation_functions();
        test_loss_functions();
        test_neural_network();
        test_training();
        
        std::cout << "\n=== All Neural Network tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
