//
// Created by rudri on 10/11/2020.

//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include "../algebra/Tensor.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = utec::algebra::Tensor<T, DIMS>;

    // Función de pérdida MSE (Mean Squared Error)
    template<typename T>
    class MSELoss : public ILoss<T, 2> {
    private:
        Tensor<T, 2> y_predicted_;
        Tensor<T, 2> y_expected_;

    public:
        // Constructor
        MSELoss(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_expected)
            : y_predicted_(y_predicted), y_expected_(y_expected) {}

        // Calcula la pérdida MSE: (1/n) * Σ(y_pred - y_true)²
        T loss() const override {
            T sum = 0;
            size_t total_elements = y_predicted_.size();
            
            for (size_t i = 0; i < y_predicted_.shape()[0]; ++i) {
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    T diff = y_predicted_(i, j) - y_expected_(i, j);
                    sum += diff * diff;
                }
            }
            
            return sum / static_cast<T>(total_elements);
        }

        // Calcula el gradiente de la pérdida: (2/n) * (y_pred - y_true)
        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> gradient(y_predicted_.shape()[0], y_predicted_.shape()[1]);
            T factor = 2.0 / static_cast<T>(y_predicted_.size());
            
            for (size_t i = 0; i < y_predicted_.shape()[0]; ++i) {
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    gradient(i, j) = factor * (y_predicted_(i, j) - y_expected_(i, j));
                }
            }
            
            return gradient;
        }
    };

    // Función de pérdida BCE (Binary Cross Entropy)
    template<typename T>
    class BCELoss : public ILoss<T, 2> {
    private:
        Tensor<T, 2> y_predicted_;
        Tensor<T, 2> y_expected_;
        static constexpr T EPSILON = 1e-7; // Valor pequeño para evitar log(0)

    public:
        // Constructor
        BCELoss(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_expected)
            : y_predicted_(y_predicted), y_expected_(y_expected) {}

        // Calcula la pérdida BCE: -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        T loss() const override {
            T sum = 0;
            size_t total_elements = y_predicted_.size();
            
            for (size_t i = 0; i < y_predicted_.shape()[0]; ++i) {
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    T pred = y_predicted_(i, j);
                    T expected = y_expected_(i, j);
                    
                    // Limitar valores para evitar log(0) o log(negativo)
                    pred = std::max(EPSILON, std::min(1.0 - EPSILON, pred));
                    
                    // BCE: -[y * log(p) + (1-y) * log(1-p)]
                    sum += -(expected * std::log(pred) + (1.0 - expected) * std::log(1.0 - pred));
                }
            }
            
            return sum / static_cast<T>(total_elements);
        }

        // Calcula el gradiente de la pérdida: (1/n) * [(y_pred - y_true) / (y_pred * (1 - y_pred))]
        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> gradient(y_predicted_.shape()[0], y_predicted_.shape()[1]);
            T factor = 1.0 / static_cast<T>(y_predicted_.size());
            
            for (size_t i = 0; i < y_predicted_.shape()[0]; ++i) {
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    T pred = y_predicted_(i, j);
                    T expected = y_expected_(i, j);
                    
                    // Limitar valores para evitar división por cero
                    pred = std::max(EPSILON, std::min(1.0 - EPSILON, pred));
                    
                    // Gradiente: (y_pred - y_true) / (y_pred * (1 - y_pred))
                    gradient(i, j) = factor * (pred - expected) / (pred * (1.0 - pred));
                }
            }
            
            return gradient;
        }
    };

    // Función de pérdida MAE (Mean Absolute Error)
    template<typename T>
    class MeanAbsoluteError {
    private:
        Tensor<T, 2> y_predicted_;
        Tensor<T, 2> y_expected_;

    public:
        MeanAbsoluteError(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_expected)
            : y_predicted_(y_predicted), y_expected_(y_expected) {}

        T compute(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) {
            T sum = 0;
            size_t total_elements = predictions.size();
            
            for (size_t i = 0; i < predictions.shape()[0]; ++i) {
                for (size_t j = 0; j < predictions.shape()[1]; ++j) {
                    sum += std::abs(predictions(i, j) - targets(i, j));
                }
            }
            
            return sum / static_cast<T>(total_elements);
        }
    };

    // Alias para compatibilidad
    template<typename T>
    using MAE = MeanAbsoluteError<T>;
    using MeanAbsoluteErrorFloat = MeanAbsoluteError<float>;

    // Función de pérdida MSE (alias para compatibilidad con tests)
    template<typename T>
    class MeanSquaredError {
    private:
        Tensor<T, 2> y_predicted_;
        Tensor<T, 2> y_expected_;

    public:
        MeanSquaredError() = default;
        
        MeanSquaredError(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_expected)
            : y_predicted_(y_predicted), y_expected_(y_expected) {}

        T compute(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) {
            T sum = 0;
            size_t total_elements = predictions.size();
            
            for (size_t i = 0; i < predictions.shape()[0]; ++i) {
                for (size_t j = 0; j < predictions.shape()[1]; ++j) {
                    T diff = predictions(i, j) - targets(i, j);
                    sum += diff * diff;
                }
            }
            
            return sum / static_cast<T>(total_elements);
        }
    };

    // Función de pérdida Cross Entropy (para clasificación multiclase)
    template<typename T>
    class CrossEntropyLoss : public ILoss<T, 2> {
    private:
        Tensor<T, 2> y_predicted_;
        Tensor<T, 2> y_expected_;
        static constexpr T EPSILON = 1e-7;

    public:
        CrossEntropyLoss(const Tensor<T, 2>& y_predicted, const Tensor<T, 2>& y_expected)
            : y_predicted_(y_predicted), y_expected_(y_expected) {}

        T loss() const override {
            T sum = 0;
            size_t batch_size = y_predicted_.shape()[0];
            T one_minus_eps = T{1} - EPSILON;
            
            for (size_t i = 0; i < batch_size; ++i) {
                T sample_loss = 0;
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    T pred = std::max(EPSILON, std::min(one_minus_eps, y_predicted_(i, j)));
                    sample_loss += y_expected_(i, j) * std::log(pred);
                }
                sum += -sample_loss;
            }
            
            return sum / static_cast<T>(batch_size);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> gradient(y_predicted_.shape()[0], y_predicted_.shape()[1]);
            T factor = -T{1} / static_cast<T>(y_predicted_.shape()[0]);
            T one_minus_eps = T{1} - EPSILON;
            
            for (size_t i = 0; i < y_predicted_.shape()[0]; ++i) {
                for (size_t j = 0; j < y_predicted_.shape()[1]; ++j) {
                    T pred = std::max(EPSILON, std::min(one_minus_eps, y_predicted_(i, j)));
                    gradient(i, j) = factor * y_expected_(i, j) / pred;
                }
            }
            
            return gradient;
        }
    };

    // Alias para compatibilidad
    template<typename T>
    using CrossEntropy = CrossEntropyLoss<T>;
    using CrossEntropyFloat = CrossEntropyLoss<float>;

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
