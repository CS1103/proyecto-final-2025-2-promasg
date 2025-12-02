//
// Created by rudri on 10/11/2020.

//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "../algebra/Tensor.h"
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <cmath>

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = utec::algebra::Tensor<T, DIMS>;
    
    // Tipos comunes para compatibilidad
    using Tensor2D = utec::algebra::Tensor<float, 2>;
    using Tensor1D = utec::algebra::Tensor<float, 1>;

    // Red neuronal
    template<typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
        
        // Función de inicialización de pesos (Xavier/Glorot)
        static void xavier_init(Tensor<T, 2>& W) {
            size_t in_features = W.shape()[1];
            size_t out_features = W.shape()[0];
            T limit = std::sqrt(6.0 / (in_features + out_features));
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(-limit, limit);
            
            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    W(i, j) = dis(gen);
                }
            }
        }
        
        // Función de inicialización de sesgos (ceros)
        static void zero_init(Tensor<T, 2>& b) {
            b.fill(T{0});
        }

    public:
        NeuralNetwork() = default;

        // Agregar una capa a la red
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.push_back(std::move(layer));
        }
        
        // Agregar una capa densa con activación (método de conveniencia)
        template<typename ActivationType>
        void add_dense_layer(size_t in_features, size_t out_features, 
                             std::shared_ptr<ActivationType> activation) {
            // Crear capa Dense
            auto dense = std::make_unique<Dense<T>>(
                in_features, out_features,
                xavier_init,
                zero_init
            );
            
            // Crear wrapper que combina Dense + Activation
            class DenseWithActivation : public ILayer<T> {
            private:
                std::unique_ptr<Dense<T>> dense_;
                std::shared_ptr<ActivationType> activation_;
                
            public:
                DenseWithActivation(std::unique_ptr<Dense<T>> dense, 
                                   std::shared_ptr<ActivationType> activation)
                    : dense_(std::move(dense)), activation_(activation) {}
                
                Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
                    Tensor<T, 2> dense_out = dense_->forward(x);
                    return activation_->forward(dense_out);
                }
                
                Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
                    Tensor<T, 2> act_grad = activation_->backward(gradients);
                    return dense_->backward(act_grad);
                }
                
                void update_params(IOptimizer<T>& optimizer) override {
                    dense_->update_params(optimizer);
                }
            };
            
            layers_.push_back(std::make_unique<DenseWithActivation>(
                std::move(dense), activation
            ));
        }

        // Forward pass: propagar entrada a través de todas las capas
        Tensor<T, 2> forward(const Tensor<T, 2>& X) {
            Tensor<T, 2> output = X;
            for (auto& layer : layers_) {
                output = layer->forward(output);
            }
            return output;
        }

        // Backward pass: propagar gradientes hacia atrás
        void backward(const Tensor<T, 2>& dL_dY) {
            Tensor<T, 2> gradient = dL_dY;
            // Propagar gradientes hacia atrás a través de las capas (en orden inverso)
            for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                gradient = layers_[i]->backward(gradient);
            }
        }

        // Entrenar la red neuronal
        template <template <typename ...> class LossType, 
                  template <typename ...> class OptimizerType = SGD>
        void train(const Tensor<T, 2>& X, const Tensor<T, 2>& Y, 
                   const size_t epochs, const size_t batch_size, T learning_rate) {
            
            // Crear optimizador
            OptimizerType<T> optimizer(learning_rate);
            
            // Entrenar por épocas
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                T total_loss = 0;
                size_t num_batches = (X.shape()[0] + batch_size - 1) / batch_size;

                // Procesar cada batch
                for (size_t batch = 0; batch < num_batches; ++batch) {
                    size_t start_idx = batch * batch_size;
                    size_t end_idx = std::min(start_idx + batch_size, X.shape()[0]);
                    size_t current_batch_size = end_idx - start_idx;

                    // Extraer batch
                    Tensor<T, 2> X_batch(current_batch_size, X.shape()[1]);
                    Tensor<T, 2> Y_batch(current_batch_size, Y.shape()[1]);
                    
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        for (size_t j = 0; j < X.shape()[1]; ++j) {
                            X_batch(i, j) = X(start_idx + i, j);
                        }
                        for (size_t j = 0; j < Y.shape()[1]; ++j) {
                            Y_batch(i, j) = Y(start_idx + i, j);
                        }
                    }

                    // Forward pass
                    Tensor<T, 2> Y_pred = forward(X_batch);

                    // Calcular pérdida
                    LossType<T> loss(Y_pred, Y_batch);
                    total_loss += loss.loss();

                    // Calcular gradiente de la pérdida
                    Tensor<T, 2> dL_dY = loss.loss_gradient();

                    // Backward pass
                    backward(dL_dY);

                    // Actualizar parámetros
                    for (auto& layer : layers_) {
                        layer->update_params(optimizer);
                    }
                }

                // Mostrar pérdida cada 100 épocas (opcional)
                // if ((epoch + 1) % 100 == 0) {
                //     std::cout << "Época " << (epoch + 1) << ", Pérdida: " << (total_loss / num_batches) << std::endl;
                // }
            }
        }

        // Realizar predicciones
        Tensor<T, 2> predict(const Tensor<T, 2>& X) {
            return forward(X);
        }
        
        // Evaluar pérdida en datos dados
        template<typename LossType>
        T evaluate(const Tensor<T, 2>& X, const Tensor<T, 2>& Y, LossType& loss) {
            Tensor<T, 2> Y_pred = forward(X);
            LossType loss_obj(Y_pred, Y);
            return loss_obj.loss();
        }
        
        // Entrenar un paso (método de conveniencia)
        template<typename LossType, typename OptimizerType>
        void train_step(const Tensor<T, 2>& X, const Tensor<T, 2>& Y, 
                       LossType& loss, OptimizerType& optimizer) {
            // Forward pass
            Tensor<T, 2> Y_pred = forward(X);
            
            // Calcular pérdida
            LossType loss_obj(Y_pred, Y);
            
            // Calcular gradiente de la pérdida
            Tensor<T, 2> dL_dY = loss_obj.loss_gradient();
            
            // Backward pass
            backward(dL_dY);
            
            // Actualizar parámetros
            for (auto& layer : layers_) {
                layer->update_params(optimizer);
            }
        }
    };
    
    // Alias para compatibilidad
    using NeuralNetworkFloat = NeuralNetwork<float>;

}

// Alias para compatibilidad
namespace utec {
    namespace nn = neural_network;
    
    // Tipos comunes
    using Tensor2D = neural_network::Tensor<float, 2>;
    using Tensor1D = neural_network::Tensor<float, 1>;
    using NeuralNetwork = neural_network::NeuralNetworkFloat;
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
