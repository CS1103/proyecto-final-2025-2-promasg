//
// Created by rudri on 10/11/2020.

//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include "../algebra/Tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = utec::algebra::Tensor<T, DIMS>;

    // Optimizador SGD (Stochastic Gradient Descent)
    template<typename T>
    class SGD final : public IOptimizer<T> {
    private:
        T learning_rate_;

    public:
        // Constructor con tasa de aprendizaje
        explicit SGD(T learning_rate = 0.01) : learning_rate_(learning_rate) {}

        // Actualizar parámetros: params = params - learning_rate * gradients
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& gradients) override {
            // Verificar que las dimensiones coincidan
            if (params.shape()[0] != gradients.shape()[0] || 
                params.shape()[1] != gradients.shape()[1]) {
                throw std::runtime_error("Las dimensiones de parámetros y gradientes no coinciden");
            }

            // Actualizar cada parámetro: param = param - lr * gradient
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    params(i, j) -= learning_rate_ * gradients(i, j);
                }
            }
        }

        // Método step() para compatibilidad con la interfaz
        void step() override {}
    };

    // Optimizador Adam (Adaptive Moment Estimation)
    template<typename T>
    class Adam final : public IOptimizer<T> {
    private:
        T learning_rate_;
        T beta1_;           // Coeficiente de decaimiento para el primer momento (media)
        T beta2_;           // Coeficiente de decaimiento para el segundo momento (varianza)
        T epsilon_;         // Valor pequeño para evitar división por cero
        size_t timestep_;   // Contador de pasos
        std::vector<Tensor<T, 2>> m_;  // Primer momento (media móvil de gradientes)
        std::vector<Tensor<T, 2>> v_;  // Segundo momento (media móvil de gradientes al cuadrado)

    public:
        // Constructor con parámetros de Adam
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), 
              epsilon_(epsilon), timestep_(0) {}

        // Actualizar parámetros usando Adam
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& gradients) override {
            // Verificar que las dimensiones coincidan
            if (params.shape()[0] != gradients.shape()[0] || 
                params.shape()[1] != gradients.shape()[1]) {
                throw std::runtime_error("Las dimensiones de parámetros y gradientes no coinciden");
            }

            // Inicializar momentos si es la primera vez
            if (m_.empty()) {
                m_.push_back(Tensor<T, 2>(params.shape()[0], params.shape()[1]));
                v_.push_back(Tensor<T, 2>(params.shape()[0], params.shape()[1]));
                m_[0].fill(0);
                v_[0].fill(0);
            }

            // Incrementar timestep
            timestep_++;
            T t = static_cast<T>(timestep_);

            // Actualizar cada parámetro
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    T g = gradients(i, j);
                    
                    // Actualizar primer momento: m = beta1 * m + (1 - beta1) * g
                    m_[0](i, j) = beta1_ * m_[0](i, j) + (T{1} - beta1_) * g;
                    
                    // Actualizar segundo momento: v = beta2 * v + (1 - beta2) * g^2
                    v_[0](i, j) = beta2_ * v_[0](i, j) + (T{1} - beta2_) * g * g;
                    
                    // Corrección de sesgo (bias correction)
                    T m_hat = m_[0](i, j) / (T{1} - std::pow(beta1_, t));
                    T v_hat = v_[0](i, j) / (T{1} - std::pow(beta2_, t));
                    
                    // Actualizar parámetro: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
                    params(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
        }

        // Método step() para compatibilidad con la interfaz
        void step() override {}
    };

    // Optimizador RMSprop (Root Mean Square Propagation)
    template<typename T>
    class RMSprop final : public IOptimizer<T> {
    private:
        T learning_rate_;
        T decay_;           // Factor de decaimiento (típicamente 0.9)
        T epsilon_;         // Valor pequeño para evitar división por cero
        std::vector<Tensor<T, 2>> cache_;  // Cache de gradientes al cuadrado

    public:
        // Constructor con parámetros de RMSprop
        explicit RMSprop(T learning_rate = 0.001, T decay = 0.9, T epsilon = 1e-8)
            : learning_rate_(learning_rate), decay_(decay), epsilon_(epsilon) {}

        // Actualizar parámetros usando RMSprop
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& gradients) override {
            // Verificar que las dimensiones coincidan
            if (params.shape()[0] != gradients.shape()[0] || 
                params.shape()[1] != gradients.shape()[1]) {
                throw std::runtime_error("Las dimensiones de parámetros y gradientes no coinciden");
            }

            // Inicializar cache si es la primera vez
            if (cache_.empty()) {
                cache_.push_back(Tensor<T, 2>(params.shape()[0], params.shape()[1]));
                cache_[0].fill(0);
            }

            // Actualizar cada parámetro
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    T g = gradients(i, j);
                    
                    // Actualizar cache: cache = decay * cache + (1 - decay) * g^2
                    cache_[0](i, j) = decay_ * cache_[0](i, j) + (T{1} - decay_) * g * g;
                    
                    // Actualizar parámetro: param = param - lr * g / (sqrt(cache) + epsilon)
                    params(i, j) -= learning_rate_ * g / (std::sqrt(cache_[0](i, j)) + epsilon_);
                }
            }
        }

        // Método step() para compatibilidad con la interfaz
        void step() override {}
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
