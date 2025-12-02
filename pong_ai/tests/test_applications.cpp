/**
 * @file test_applications.cpp
 * @brief Casos de prueba para las aplicaciones
 */

#include "../include/utec/apps/PatternClassifier.h"
#include "../include/utec/apps/SequencePredictor.h"
#include "../include/utec/apps/ControllerDemo.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace utec::apps;

void test_pattern_classifier() {
    std::cout << "Testing PatternClassifier..." << std::endl;
    
    PatternClassifier classifier;
    
    // Generate training data
    auto [train_inputs, train_targets] = classifier.generate_training_data(300);
    assert(train_inputs.shape()[0] == 300);
    assert(train_inputs.shape()[1] == 4);
    assert(train_targets.shape()[0] == 300);
    assert(train_targets.shape()[1] == 3);
    
    // Train the classifier
    classifier.train(50, 32, 0.01f);
    
    // Test classification
    auto [test_inputs, test_targets] = classifier.generate_training_data(100);
    float accuracy = classifier.evaluate(test_inputs, test_targets);
    
    std::cout << "Classification accuracy: " << (accuracy * 100) << "%" << std::endl;
    assert(accuracy > 0.0f);  // Should have some accuracy
    
    std::cout << "✓ PatternClassifier tests passed" << std::endl;
}

void test_sequence_predictor() {
    std::cout << "Testing SequencePredictor..." << std::endl;
    
    SequencePredictor predictor;
    
    // Generate a simple linear sequence
    auto sequence = predictor.generate_sequence(50, 0);  // Linear sequence
    assert(sequence.size() == 50);
    
    // Prepare training data
    auto [train_inputs, train_targets] = predictor.prepare_training_data(sequence, 5);
    assert(train_inputs.shape()[0] == 45);  // 50 - 5
    assert(train_inputs.shape()[1] == 5);
    assert(train_targets.shape()[0] == 45);
    assert(train_targets.shape()[1] == 1);
    
    // Train the predictor
    predictor.train(sequence, 50, 5, 0.01f);
    
    // Test prediction
    utec::Tensor2D history(1, 5);
    for (size_t i = 0; i < 5; ++i) {
        history(0, i) = sequence[i];
    }
    
    float prediction = predictor.predict_next(history);
    std::cout << "Next value prediction: " << prediction << std::endl;
    
    // Test multi-step prediction
    auto predictions = predictor.predict_ahead(history, 5);
    assert(predictions.size() == 5);
    
    std::cout << "✓ SequencePredictor tests passed" << std::endl;
}

void test_controller_demo() {
    std::cout << "Testing ControllerDemo..." << std::endl;
    
    ControllerDemo controller;
    
    // Generate training data
    auto [train_states, train_actions] = controller.generate_training_data(20, 10);
    assert(train_states.shape()[0] == 200);  // 20 episodes * 10 steps
    assert(train_states.shape()[1] == 4);
    assert(train_actions.shape()[0] == 200);
    assert(train_actions.shape()[1] == 5);
    
    // Train the controller
    controller.train(50, 32, 0.01f);
    
    // Test decision making
    ControllerDemo::SystemState state;
    state.position = 0.0f;
    state.velocity = 0.0f;
    state.target_pos = 0.5f;
    state.time_step = 0.0f;
    
    auto action = controller.decide_action(state);
    float confidence = controller.get_action_confidence(state);
    
    std::cout << "Decided action: " << static_cast<int>(action) 
              << " with confidence: " << confidence << std::endl;
    assert(confidence >= 0.0f && confidence <= 1.0f);
    
    // Test simulation
    auto result = controller.simulate(20, 0.8f);
    assert(result.positions.size() == 20);
    assert(result.velocities.size() == 20);
    assert(result.actions.size() == 20);
    assert(result.final_error >= 0.0f);
    
    std::cout << "Simulation final error: " << result.final_error << std::endl;
    
    // Test evaluation
    float avg_error = controller.evaluate(5, 20);
    std::cout << "Average evaluation error: " << avg_error << std::endl;
    assert(avg_error >= 0.0f);
    
    std::cout << "✓ ControllerDemo tests passed" << std::endl;
}

int main() {
    std::cout << "=== Running Application Tests ===" << std::endl;
    
    try {
        test_pattern_classifier();
        test_sequence_predictor();
        test_controller_demo();
        
        std::cout << "\n=== All Application tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
