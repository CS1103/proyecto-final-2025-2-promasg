/**
 * @file SequencePredictor.cpp
 * @brief Implementación del predictor de series numéricas
 */

#include "utec/apps/SequencePredictor.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace utec::apps {

SequencePredictor::SequencePredictor() {
    // Construir la red neuronal: window_size -> HIDDEN_UNITS -> 1
    network_.add_dense_layer(5, HIDDEN_UNITS, std::make_shared<nn::ReLU<float>>());
    network_.add_dense_layer(HIDDEN_UNITS, 1, std::make_shared<nn::Linear<float>>());
}

std::vector<float> SequencePredictor::generate_sequence(size_t length, int sequence_type) {
    std::vector<float> sequence;
    sequence.reserve(length);
    
    for (size_t i = 0; i < length; ++i) {
        float value = 0.0f;
        
        switch (sequence_type) {
            case 0:  // Lineal: y = 2x + 1
                value = 2.0f * static_cast<float>(i) + 1.0f;
                break;
            case 1:  // Cuadrática: y = x^2
                value = static_cast<float>(i * i);
                break;
            case 2:  // Senoidal: y = sin(x/10)
                value = std::sin(static_cast<float>(i) / 10.0f);
                break;
            default:
                value = static_cast<float>(i);
        }
        
        sequence.push_back(value);
    }
    
    return sequence;
}

std::pair<nn::Tensor2D, nn::Tensor2D> SequencePredictor::prepare_training_data(
    const std::vector<float>& sequence, size_t window_size) {
    
    if (sequence.size() <= window_size) {
        throw std::invalid_argument("secuencia demasiada corta");
    }
    
    size_t num_samples = sequence.size() - window_size;
    nn::Tensor2D inputs(num_samples, window_size);
    nn::Tensor2D targets(num_samples, 1);
    
    for (size_t i = 0; i < num_samples; ++i) {
        // Input: ventana de valores anteriores
        for (size_t j = 0; j < window_size; ++j) {
            inputs(i, j) = sequence[i + j];
        }
        // Target: siguiente valor
        targets(i, 0) = sequence[i + window_size];
    }
    
    return {inputs, targets};
}

void SequencePredictor::train(const std::vector<float>& sequence,
                              size_t epochs, size_t window_size, float learning_rate) {
    std::cout << "Entrenando predictor de series..." << std::endl;
    
    // Preparar datos
    auto [train_inputs, train_targets] = prepare_training_data(sequence, window_size);
    
    // Normalizar datos (opcional, pero ayuda con el entrenamiento)
    // Por simplicidad, asumimos que los datos ya están en un rango razonable
    
    // Crear pérdida y optimizador
    nn::MSELoss<float> loss_obj(train_inputs, train_targets);
    nn::Adam<float> optimizer(learning_rate);
    
    // Entrenar
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        network_.train_step(train_inputs, train_targets, loss_obj, optimizer);
        
        if ((epoch + 1) % 20 == 0) {
            float current_loss = network_.evaluate(train_inputs, train_targets, loss_obj);
            std::cout << "Época " << (epoch + 1) << "/" << epochs 
                      << ", Pérdida: " << current_loss << std::endl;
        }
    }
}

float SequencePredictor::predict_next(const nn::Tensor2D& history) {
    nn::Tensor2D output = network_.forward(history);
    return output(0, 0);
}

std::vector<float> SequencePredictor::predict_ahead(const nn::Tensor2D& history, size_t steps) {
    std::vector<float> predictions;
    predictions.reserve(steps);
    
    // Usar la historia inicial
    nn::Tensor2D current_window = history;
    
    for (size_t step = 0; step < steps; ++step) {
        // Predecir siguiente valor
        float next_val = predict_next(current_window);
        predictions.push_back(next_val);
        
        // Actualizar ventana: desplazar y agregar predicción
        if (current_window.shape()[1] > 1) {
            for (size_t j = 0; j < current_window.shape()[1] - 1; ++j) {
                current_window(0, j) = current_window(0, j + 1);
            }
            current_window(0, current_window.shape()[1] - 1) = next_val;
        }
    }
    
    return predictions;
}

float SequencePredictor::evaluate(const std::vector<float>& test_sequence, size_t window_size) {
    if (test_sequence.size() <= window_size) {
        return 0.0f;
    }
    
    auto [test_inputs, test_targets] = prepare_training_data(test_sequence, window_size);
    
    nn::MSELoss<float> loss_obj(test_inputs, test_targets);
    return network_.evaluate(test_inputs, test_targets, loss_obj);
}

void SequencePredictor::save_model(const std::string& filename) {
    // TODO: Implementar serialización
    (void)filename;  // Evitar warning
}

void SequencePredictor::load_model(const std::string& filename) {
    // TODO: Implementar deserialización
    (void)filename;  // Evitar warning
}

} // namespace utec::apps
