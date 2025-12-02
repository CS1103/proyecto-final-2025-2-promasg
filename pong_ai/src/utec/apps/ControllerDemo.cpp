/**
 * @file ControllerDemo.cpp
 * @brief Implementación del controlador de sistema
 */

#include "utec/apps/ControllerDemo.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace utec::apps {

ControllerDemo::ControllerDemo() {
    // Construir la red neuronal: 4 -> 16 -> 5
    network_.add_dense_layer(STATE_FEATURES, 16, std::make_shared<nn::ReLU<float>>());
    network_.add_dense_layer(16, NUM_ACTIONS, std::make_shared<nn::Softmax<float>>());
}

std::pair<nn::Tensor2D, nn::Tensor2D> ControllerDemo::generate_training_data(
    size_t num_episodes, size_t steps_per_episode) {
    
    size_t total_samples = num_episodes * steps_per_episode;
    nn::Tensor2D states(total_samples, STATE_FEATURES);
    nn::Tensor2D actions(total_samples, NUM_ACTIONS);
    
    actions.fill(0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> target_dis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> pos_dis(-0.5f, 0.5f);
    std::uniform_real_distribution<float> vel_dis(-0.3f, 0.3f);
    
    size_t idx = 0;
    
    for (size_t episode = 0; episode < num_episodes; ++episode) {
        float target = target_dis(gen);
        SystemState state;
        state.position = pos_dis(gen);
        state.velocity = vel_dis(gen);
        state.target_pos = target;
        
        for (size_t step = 0; step < steps_per_episode && idx < total_samples; ++step, ++idx) {
            state.time_step = static_cast<float>(step) / steps_per_episode;
            
            // Convertir estado a tensor
            nn::Tensor2D state_tensor = state_to_tensor(state);
            for (size_t j = 0; j < STATE_FEATURES; ++j) {
                states(idx, j) = state_tensor(0, j);
            }
            
            // Decidir acción basada en regla simple (para generar datos de entrenamiento)
            Action action = Action::STAY;
            float error = state.target_pos - state.position;
            
            if (std::abs(error) > 0.1f) {
                if (error > 0) {
                    action = Action::MOVE_RIGHT;
                } else {
                    action = Action::MOVE_LEFT;
                }
            } else if (std::abs(state.velocity) > 0.1f) {
                if (state.velocity > 0) {
                    action = Action::DECELERATE;
                } else {
                    action = Action::ACCELERATE;
                }
            }
            
            // One-hot encoding de la acción
            actions(idx, static_cast<size_t>(action)) = 1.0f;
            
            // Aplicar acción y actualizar estado
            state = apply_action(state, action);
        }
    }
    
    return {states, actions};
}

void ControllerDemo::train(size_t epochs, size_t batch_size, float learning_rate) {
    std::cout << "Entrenando controlador..." << std::endl;
    
    // Generar datos de entrenamiento
    auto [train_states, train_actions] = generate_training_data(20, 10);
    
    // Crear pérdida y optimizador
    nn::CrossEntropyLoss<float> loss_obj(train_states, train_actions);
    nn::Adam<float> optimizer(learning_rate);
    
    // Entrenar
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        network_.train_step(train_states, train_actions, loss_obj, optimizer);
        
        if ((epoch + 1) % 20 == 0) {
            float current_loss = network_.evaluate(train_states, train_actions, loss_obj);
            std::cout << "Época " << (epoch + 1) << "/" << epochs 
                      << ", Pérdida: " << current_loss << std::endl;
        }
    }
}

ControllerDemo::Action ControllerDemo::decide_action(const SystemState& state) {
    nn::Tensor2D state_tensor = state_to_tensor(state);
    nn::Tensor2D output = network_.forward(state_tensor);
    
    // Encontrar la acción con mayor probabilidad
    size_t max_idx = 0;
    float max_val = output(0, 0);
    
    for (size_t j = 1; j < NUM_ACTIONS; ++j) {
        if (output(0, j) > max_val) {
            max_val = output(0, j);
            max_idx = j;
        }
    }
    
    return static_cast<Action>(max_idx);
}

float ControllerDemo::get_action_confidence(const SystemState& state) {
    nn::Tensor2D state_tensor = state_to_tensor(state);
    nn::Tensor2D output = network_.forward(state_tensor);
    
    // Retornar la máxima probabilidad
    float max_val = output(0, 0);
    for (size_t j = 1; j < NUM_ACTIONS; ++j) {
        if (output(0, j) > max_val) {
            max_val = output(0, j);
        }
    }
    
    return max_val;
}

ControllerDemo::SimulationResult ControllerDemo::simulate(size_t num_steps, float target_position) {
    SimulationResult result;
    
    SystemState state;
    state.position = 0.0f;
    state.velocity = 0.0f;
    state.target_pos = target_position;
    
    for (size_t step = 0; step < num_steps; ++step) {
        state.time_step = static_cast<float>(step) / num_steps;
        
        result.positions.push_back(state.position);
        result.velocities.push_back(state.velocity);
        
        Action action = decide_action(state);
        result.actions.push_back(static_cast<int>(action));
        
        state = apply_action(state, action);
    }
    
    result.final_error = std::abs(state.position - target_position);
    return result;
}

float ControllerDemo::evaluate(size_t num_simulations, size_t num_steps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    float total_error = 0.0f;
    
    for (size_t sim = 0; sim < num_simulations; ++sim) {
        float target = dis(gen);
        auto result = simulate(num_steps, target);
        total_error += result.final_error;
    }
    
    return total_error / num_simulations;
}

void ControllerDemo::save_model(const std::string& filename) {
    // TODO: Implementar serialización
    (void)filename;  // Evitar warning
}

void ControllerDemo::load_model(const std::string& filename) {
    // TODO: Implementar deserialización
    (void)filename;  // Evitar warning
}

nn::Tensor2D ControllerDemo::state_to_tensor(const SystemState& state) {
    nn::Tensor2D tensor(1, STATE_FEATURES);
    tensor(0, 0) = state.position;
    tensor(0, 1) = state.velocity;
    tensor(0, 2) = state.target_pos;
    tensor(0, 3) = state.time_step;
    return tensor;
}

ControllerDemo::SystemState ControllerDemo::apply_action(const SystemState& state, Action action) {
    SystemState new_state = state;
    
    switch (action) {
        case Action::MOVE_LEFT:
            new_state.position -= 0.1f;
            break;
        case Action::MOVE_RIGHT:
            new_state.position += 0.1f;
            break;
        case Action::ACCELERATE:
            new_state.velocity += 0.1f;
            break;
        case Action::DECELERATE:
            new_state.velocity -= 0.1f;
            break;
        case Action::STAY:
        default:
            break;
    }
    
    // Limitar posición y velocidad
    new_state.position = std::clamp(new_state.position, -1.0f, 1.0f);
    new_state.velocity = std::clamp(new_state.velocity, -1.0f, 1.0f);
    
    return new_state;
}

} // namespace utec::apps
