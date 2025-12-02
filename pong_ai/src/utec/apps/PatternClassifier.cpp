/**
 * @file PatternClassifier.cpp
 * @brief Implementación del clasificador de patrones
 */

#include "utec/apps/PatternClassifier.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace utec::apps {

PatternClassifier::PatternClassifier() {
    // Construir la red neuronal: 4 -> 16 -> 3
    network_.add_dense_layer(INPUT_FEATURES, 16, std::make_shared<nn::ReLU<float>>());
    network_.add_dense_layer(16, OUTPUT_CLASSES, std::make_shared<nn::Softmax<float>>());
}

std::pair<nn::Tensor2D, nn::Tensor2D> PatternClassifier::generate_training_data(size_t num_samples) {
    size_t samples_per_class = num_samples / 3;
    nn::Tensor2D inputs(num_samples, INPUT_FEATURES);
    nn::Tensor2D targets(num_samples, OUTPUT_CLASSES);
    
    targets.fill(0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dis(-0.1f, 0.1f);
    
    size_t idx = 0;
    
    // Generar círculos
    for (size_t i = 0; i < samples_per_class && idx < num_samples; ++i, ++idx) {
        auto sample = generate_circle_sample();
        for (size_t j = 0; j < INPUT_FEATURES; ++j) {
            inputs(idx, j) = sample(0, j) + noise_dis(gen);
        }
        targets(idx, static_cast<size_t>(Pattern::CIRCLE)) = 1.0f;
    }
    
    // Generar cuadrados
    for (size_t i = 0; i < samples_per_class && idx < num_samples; ++i, ++idx) {
        auto sample = generate_square_sample();
        for (size_t j = 0; j < INPUT_FEATURES; ++j) {
            inputs(idx, j) = sample(0, j) + noise_dis(gen);
        }
        targets(idx, static_cast<size_t>(Pattern::SQUARE)) = 1.0f;
    }
    
    // Generar triángulos
    for (size_t i = 0; i < samples_per_class && idx < num_samples; ++i, ++idx) {
        auto sample = generate_triangle_sample();
        for (size_t j = 0; j < INPUT_FEATURES; ++j) {
            inputs(idx, j) = sample(0, j) + noise_dis(gen);
        }
        targets(idx, static_cast<size_t>(Pattern::TRIANGLE)) = 1.0f;
    }
    
    return {inputs, targets};
}

void PatternClassifier::train(size_t epochs, size_t batch_size, float learning_rate) {
    std::cout << "Entrenando clasificador de patrones..." << std::endl;
    
    // Generar datos de entrenamiento
    auto [train_inputs, train_targets] = generate_training_data(300);
    
    // Crear pérdida y optimizador
    nn::CrossEntropyLoss<float> loss_obj(train_inputs, train_targets);
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

PatternClassifier::Pattern PatternClassifier::classify(const nn::Tensor2D& input) {
    nn::Tensor2D output = network_.forward(input);
    
    // Encontrar la clase con mayor probabilidad
    size_t max_idx = 0;
    float max_val = output(0, 0);
    
    for (size_t j = 1; j < OUTPUT_CLASSES; ++j) {
        if (output(0, j) > max_val) {
            max_val = output(0, j);
            max_idx = j;
        }
    }
    
    return static_cast<Pattern>(max_idx);
}

float PatternClassifier::get_confidence(const nn::Tensor2D& input) {
    nn::Tensor2D output = network_.forward(input);
    
    // Retornar la máxima probabilidad
    float max_val = output(0, 0);
    for (size_t j = 1; j < OUTPUT_CLASSES; ++j) {
        if (output(0, j) > max_val) {
            max_val = output(0, j);
        }
    }
    
    return max_val;
}

float PatternClassifier::evaluate(const nn::Tensor2D& test_inputs, const nn::Tensor2D& test_targets) {
    size_t correct = 0;
    size_t total = test_inputs.shape()[0];
    
    for (size_t i = 0; i < total; ++i) {
        // Extraer muestra individual
        nn::Tensor2D sample(1, INPUT_FEATURES);
        for (size_t j = 0; j < INPUT_FEATURES; ++j) {
            sample(0, j) = test_inputs(i, j);
        }
        
        // Clasificar
        Pattern predicted = classify(sample);
        
        // Verificar si es correcto
        size_t true_class = 0;
        for (size_t j = 0; j < OUTPUT_CLASSES; ++j) {
            if (test_targets(i, j) > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        if (static_cast<size_t>(predicted) == true_class) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / static_cast<float>(total);
}

void PatternClassifier::save_model(const std::string& filename) {
    // TODO: Implementar serialización
    (void)filename;  // Evitar warning
}

void PatternClassifier::load_model(const std::string& filename) {
    // TODO: Implementar deserialización
    (void)filename;  // Evitar warning
}

nn::Tensor2D PatternClassifier::generate_circle_sample() const {
    // Círculo: área alta, perímetro medio, compactness alta (~1.0), aspect_ratio ~1.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> area_dis(0.7f, 1.0f);
    std::uniform_real_distribution<float> perimeter_dis(0.5f, 0.8f);
    
    nn::Tensor2D sample(1, INPUT_FEATURES);
    sample(0, 0) = area_dis(gen);           // area
    sample(0, 1) = perimeter_dis(gen);      // perimeter
    sample(0, 2) = 0.9f + (gen() % 10) * 0.01f;  // compactness (cerca de 1.0)
    sample(0, 3) = 0.9f + (gen() % 10) * 0.01f;  // aspect_ratio (cerca de 1.0)
    
    return sample;
}

nn::Tensor2D PatternClassifier::generate_square_sample() const {
    // Cuadrado: área media, perímetro alto, compactness media, aspect_ratio ~1.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> area_dis(0.4f, 0.7f);
    std::uniform_real_distribution<float> perimeter_dis(0.7f, 1.0f);
    
    nn::Tensor2D sample(1, INPUT_FEATURES);
    sample(0, 0) = area_dis(gen);           // area
    sample(0, 1) = perimeter_dis(gen);      // perimeter
    sample(0, 2) = 0.7f + (gen() % 20) * 0.01f;  // compactness
    sample(0, 3) = 0.9f + (gen() % 10) * 0.01f;  // aspect_ratio (cerca de 1.0)
    
    return sample;
}

nn::Tensor2D PatternClassifier::generate_triangle_sample() const {
    // Triángulo: área baja, perímetro medio-alto, compactness baja, aspect_ratio variable
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> area_dis(0.2f, 0.5f);
    std::uniform_real_distribution<float> perimeter_dis(0.6f, 0.9f);
    
    nn::Tensor2D sample(1, INPUT_FEATURES);
    sample(0, 0) = area_dis(gen);           // area
    sample(0, 1) = perimeter_dis(gen);      // perimeter
    sample(0, 2) = 0.5f + (gen() % 30) * 0.01f;  // compactness (baja)
    sample(0, 3) = 0.6f + (gen() % 40) * 0.01f;  // aspect_ratio (variable)
    
    return sample;
}

} // namespace utec::apps
