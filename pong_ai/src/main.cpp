/**
 * @file main.cpp
 * @brief Demostración de las aplicaciones de la red neuronal
 */

#include "utec/apps/PatternClassifier.h"
#include "utec/apps/SequencePredictor.h"
#include "utec/apps/ControllerDemo.h"
#include <iostream>
#include <iomanip>

using namespace utec::apps;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void demo_pattern_classifier() {
    print_separator("DEMO 1: Pattern Classifier");
    
    std::cout << "\nEntrenando clasificador de patrones geométricos..." << std::endl;
    std::cout << "Patrones: Círculos, Cuadrados, Triángulos\n" << std::endl;
    
    PatternClassifier classifier;
    classifier.train(100, 32, 0.01f);
    
    // Generate test data
    auto [test_inputs, test_targets] = classifier.generate_training_data(100);
    float accuracy = classifier.evaluate(test_inputs, test_targets);
    
    std::cout << "\nEntrenamiento completado" << std::endl;
    std::cout << "Precisión en datos de prueba: " << std::fixed << std::setprecision(2) 
              << (accuracy * 100) << "%" << std::endl;
    
    // Test classification on a single sample
    std::cout << "\nClasificando muestras individuales:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        utec::Tensor2D sample(1, 4);
        for (size_t j = 0; j < 4; ++j) {
            sample(0, j) = test_inputs(i, j);
        }
        
        auto pattern = classifier.classify(sample);
        float confidence = classifier.get_confidence(sample);
        
        std::string pattern_name;
        switch (pattern) {
            case PatternClassifier::Pattern::CIRCLE:
                pattern_name = "Círculo";
                break;
            case PatternClassifier::Pattern::SQUARE:
                pattern_name = "Cuadrado";
                break;
            case PatternClassifier::Pattern::TRIANGLE:
                pattern_name = "Triángulo";
                break;
        }
        
        std::cout << "  Muestra " << (i + 1) << ": " << pattern_name 
                  << " (confianza: " << std::fixed << std::setprecision(2) 
                  << (confidence * 100) << "%)" << std::endl;
    }
}

void demo_sequence_predictor() {
    print_separator("DEMO 2: Sequence Predictor");
    
    std::cout << "\nEntrenando predictor de series numéricas..." << std::endl;
    std::cout << "Tipo de secuencia: Lineal (y = 2x + 1)\n" << std::endl;
    
    SequencePredictor predictor;
    
    // Generate and train on a linear sequence
    auto sequence = predictor.generate_sequence(100, 0);  // Linear
    predictor.train(sequence, 100, 5, 0.01f);
    
    std::cout << "\n✓ Entrenamiento completado" << std::endl;
    
    // Test prediction
    std::cout << "\nPrediciendo próximos valores:" << std::endl;
    utec::Tensor2D history(1, 5);
    for (size_t i = 0; i < 5; ++i) {
        history(0, i) = sequence[i];
    }
    
    std::cout << "Historial: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(1) << history(0, i) << " ";
    }
    std::cout << std::endl;
    
    auto predictions = predictor.predict_ahead(history, 5);
    std::cout << "Predicciones: ";
    for (float pred : predictions) {
        std::cout << std::fixed << std::setprecision(1) << pred << " ";
    }
    std::cout << std::endl;
    
    // Evaluate on test sequence
    auto test_sequence = predictor.generate_sequence(50, 0);
    float mse = predictor.evaluate(test_sequence, 5);
    std::cout << "\nError cuadrático medio (MSE): " << std::fixed << std::setprecision(4) 
              << mse << std::endl;
}

void demo_controller() {
    print_separator("DEMO 3: Controller Demo");
    
    std::cout << "\nEntrenando controlador de sistema..." << std::endl;
    std::cout << "Objetivo: Mover desde posición inicial a posición objetivo\n" << std::endl;
    
    ControllerDemo controller;
    controller.train(100, 32, 0.01f);
    
    std::cout << "\n✓ Entrenamiento completado" << std::endl;
    
    // Simulate the controller
    std::cout << "\nSimulando controlador en acción:" << std::endl;
    float target = 0.8f;
    auto result = controller.simulate(15, target);
    
    std::cout << "Posición objetivo: " << std::fixed << std::setprecision(2) << target << std::endl;
    std::cout << "\nHistorial de simulación:" << std::endl;
    std::cout << std::setw(6) << "Paso" << std::setw(12) << "Posición" 
              << std::setw(12) << "Velocidad" << std::setw(10) << "Acción" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    std::string action_names[] = {"Izq", "Stay", "Der", "Acel", "Decel"};
    
    for (size_t i = 0; i < result.positions.size(); ++i) {
        std::cout << std::setw(6) << i 
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.positions[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.velocities[i]
                  << std::setw(10) << action_names[result.actions[i]] << std::endl;
    }
    
    std::cout << "\nError final: " << std::fixed << std::setprecision(4) 
              << result.final_error << std::endl;
    
    // Evaluate controller performance
    float avg_error = controller.evaluate(10, 20);
    std::cout << "Error promedio en 10 simulaciones: " << std::fixed << std::setprecision(4) 
              << avg_error << std::endl;
}

void print_project_info() {
    print_separator("PONG AI - Red Neuronal en C++20");
    
    std::cout << "\nEpic 1: Biblioteca de Álgebra" << std::endl;
    std::cout << "  Tensor<T, Rank> genérico con acceso variádico" << std::endl;
    std::cout << "  Operaciones elemento a elemento" << std::endl;
    std::cout << "  Almacenamiento contiguo en memoria" << std::endl;
    
    std::cout << "\nEpic 2: Red Neuronal Full-Stack" << std::endl;
    std::cout << "  Capas densas (fully connected)" << std::endl;
    std::cout << "  Funciones de activación: ReLU, Sigmoid, Tanh, Linear, Softmax" << std::endl;
    std::cout << "  Funciones de pérdida: MSE, CrossEntropy, BinaryCrossEntropy, MAE" << std::endl;
    std::cout << "  Optimizadores: SGD, Adam, RMSprop" << std::endl;
    std::cout << "  Forward/Backward pass completo" << std::endl;
    std::cout << "  Serialización de modelos" << std::endl;
    
    std::cout << "\nEpic 3: Aplicaciones Prácticas" << std::endl;
    std::cout << "  PatternClassifier: Clasificación de patrones geométricos" << std::endl;
    std::cout << "  SequencePredictor: Predicción de series numéricas" << std::endl;
    std::cout << "  ControllerDemo: Control de sistema" << std::endl;
    
    std::cout << "\nCaracterísticas:" << std::endl;
    std::cout << "  C++20 moderno" << std::endl;
    std::cout << "  Código header-only para Tensor" << std::endl;
    std::cout << "  Patrones de diseño (Strategy, Factory, Template)" << std::endl;
    std::cout << "  Casos de prueba automatizados" << std::endl;
    std::cout << "  Documentación completa" << std::endl;
}

int main() {
    try {
        print_project_info();
        
        // Run demos
        demo_pattern_classifier();
        demo_sequence_predictor();
        demo_controller();
        
        print_separator("DEMOSTRACIÓN COMPLETADA");
        std::cout << "\nTodos los demos se ejecutaron exitosamente\n" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\n Error: " << e.what() << std::endl;
        return 1;
    }
}
