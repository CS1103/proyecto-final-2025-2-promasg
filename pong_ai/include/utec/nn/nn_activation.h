//
// Created by rudri on 10/11/2020.

//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include "../algebra/Tensor.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

// ReLU Activation Function: max(0, x)
template<typename T>
class ReLU : public ILayer<T> {
private:
    Tensor<T, 2> input_;

public:
    ReLU() = default;

    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        input_ = x;
        Tensor<T, 2> output(x.shape()[0], x.shape()[1]);
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                output(i, j) = std::max(T{0}, x(i, j));
            }
        }
        return output;
    }
    
    // Método de compatibilidad para tests
    Tensor<T, 2> activate(const Tensor<T, 2>& x) {
        return forward(x);
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        Tensor<T, 2> output(gradients.shape()[0], gradients.shape()[1]);
        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                output(i, j) = gradients(i, j) * (input_(i, j) > T{0} ? T{1} : T{0});
            }
        }
        return output;
    }
};

// Sigmoid Activation Function: 1 / (1 + exp(-x))
template<typename T>
class Sigmoid : public ILayer<T> {
private:
    Tensor<T, 2> sigmoid_output_;

public:
    Sigmoid() = default;

    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        sigmoid_output_ = Tensor<T, 2>(x.shape()[0], x.shape()[1]);
        Tensor<T, 2> output(x.shape()[0], x.shape()[1]);
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                // Evitar overflow: sigmoid(x) = 1 / (1 + exp(-x))
                T val = x(i, j);
                T sig;
                if (val > T{0}) {
                    sig = T{1} / (T{1} + std::exp(-val));
                } else {
                    T exp_val = std::exp(val);
                    sig = exp_val / (T{1} + exp_val);
                }
                sigmoid_output_(i, j) = sig;
                output(i, j) = sig;
            }
        }
        return output;
    }
    
    // Método de compatibilidad para tests
    Tensor<T, 2> activate(const Tensor<T, 2>& x) {
        return forward(x);
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        Tensor<T, 2> output(gradients.shape()[0], gradients.shape()[1]);
        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                T sig = sigmoid_output_(i, j);
                output(i, j) = gradients(i, j) * sig * (T{1} - sig);
            }
        }
        return output;
    }
};

// Tanh Activation Function: (e^x - e^(-x)) / (e^x + e^(-x))
template<typename T>
class Tanh : public ILayer<T> {
private:
    Tensor<T, 2> tanh_output_;

public:
    Tanh() = default;

    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        tanh_output_ = Tensor<T, 2>(x.shape()[0], x.shape()[1]);
        Tensor<T, 2> output(x.shape()[0], x.shape()[1]);
        
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                T val = x(i, j);
                T tanh_val = std::tanh(val);
                tanh_output_(i, j) = tanh_val;
                output(i, j) = tanh_val;
            }
        }
        return output;
    }
    
    // Método de compatibilidad para tests
    Tensor<T, 2> activate(const Tensor<T, 2>& x) {
        return forward(x);
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        Tensor<T, 2> output(gradients.shape()[0], gradients.shape()[1]);
        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                T tanh_val = tanh_output_(i, j);
                // Derivada: 1 - tanh^2(x)
                output(i, j) = gradients(i, j) * (T{1} - tanh_val * tanh_val);
            }
        }
        return output;
    }
};

// Linear Activation Function (Identity): f(x) = x
template<typename T>
class Linear : public ILayer<T> {
public:
    Linear() = default;

    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        return x;  // Retorna la entrada sin cambios
    }
    
    // Método de compatibilidad para tests
    Tensor<T, 2> activate(const Tensor<T, 2>& x) {
        return forward(x);
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        return gradients;  // Derivada es 1, retorna gradientes sin cambios
    }
};

// Softmax Activation Function: e^x_i / sum(e^x_j)
template<typename T>
class Softmax : public ILayer<T> {
private:
    Tensor<T, 2> softmax_output_;

public:
    Softmax() = default;

    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        softmax_output_ = Tensor<T, 2>(x.shape()[0], x.shape()[1]);
        Tensor<T, 2> output(x.shape()[0], x.shape()[1]);
        
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            // Encontrar máximo para estabilidad numérica
            T max_val = x(i, 0);
            for (size_t j = 1; j < x.shape()[1]; ++j) {
                if (x(i, j) > max_val) max_val = x(i, j);
            }
            
            // Calcular exp(x - max)
            T sum_exp = 0;
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                T exp_val = std::exp(x(i, j) - max_val);
                output(i, j) = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalizar
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                output(i, j) /= sum_exp;
                softmax_output_(i, j) = output(i, j);
            }
        }
        return output;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        Tensor<T, 2> output(gradients.shape()[0], gradients.shape()[1]);
        
        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                T softmax_j = softmax_output_(i, j);
                T grad_sum = 0;
                
                for (size_t k = 0; k < gradients.shape()[1]; ++k) {
                    T softmax_k = softmax_output_(i, k);
                    T delta_jk = (j == k) ? T{1} : T{0};
                    grad_sum += gradients(i, k) * softmax_k * (delta_jk - softmax_j);
                }
                output(i, j) = grad_sum;
            }
        }
        return output;
    }
};

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
