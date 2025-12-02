/**
 * @file example_usage.cpp
 * @brief Ejemplo de uso de la red neuronal basado en el README
 */

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace utec::nn;

int main() {
    std::cout << "Ejemplo de Uso de Red Neuronal ===" << std::endl;
    std::cout << std::endl;
    
    // Crear red
    utec::NeuralNetwork nn;
    nn.add_dense_layer(4, 8, std::make_shared<ReLU<float>>());
    nn.add_dense_layer(8, 2, std::make_shared<Softmax<float>>());
    
    std::cout << "Red neuronal creada: 4 -> 8 (ReLU) -> 2 (Softmax)" << std::endl;
    
    // Generar datos de ejemplo
    Tensor2D input(2, 4);
    Tensor2D targets(2, 2);
    
    // Llenar con datos aleatorios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            input(i, j) = dis(gen);
        }
        // One-hot encoding para targets
        targets(i, 0) = (i == 0) ? 1.0f : 0.0f;
        targets(i, 1) = (i == 1) ? 1.0f : 0.0f;
    }
    
    std::cout << "Datos generados: 2 muestras con 4 características" << std::endl;
    std::cout << std::endl;
    
    // Entrenar
    std::cout << "Entrenando red neuronal..." << std::endl;
    CrossEntropyLoss<float> loss_obj(input, targets);
    Adam<float> optimizer(0.001f);
    
    float initial_loss = nn.evaluate(input, targets, loss_obj);
    std::cout << "Pérdida inicial: " << initial_loss << std::endl;
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        nn.train_step(input, targets, loss_obj, optimizer);
        
        if ((epoch + 1) % 25 == 0) {
            float current_loss = nn.evaluate(input, targets, loss_obj);
            std::cout << "Época " << (epoch + 1) << "/100, Pérdida: " << current_loss << std::endl;
        }
    }
    
    float final_loss = nn.evaluate(input, targets, loss_obj);
    std::cout << "Pérdida final: " << final_loss << std::endl;
    std::cout << "Reducción de pérdida: " << ((initial_loss - final_loss) / initial_loss * 100) << "%" << std::endl;
    std::cout << std::endl;
    
    // Realizar predicciones
    std::cout << "Realizando predicciones..." << std::endl;
    Tensor2D predictions = nn.predict(input);
    
    std::cout << "Predicciones (probabilidades):" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "  Muestra " << (i + 1) << ": ";
        std::cout << "Clase 0: " << std::fixed << std::setprecision(3) << predictions(i, 0) 
                  << ", Clase 1: " << predictions(i, 1) << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Ejemplo completado!" << std::endl;
    
    return 0;
}

