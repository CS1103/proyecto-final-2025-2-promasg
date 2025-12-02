//
// Created by rudri on 10/11/2020.

//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "../algebra/Tensor.h"

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = utec::algebra::Tensor<T, DIMS>;

    // Función auxiliar para multiplicación de matrices: C = A @ B^T
    template<typename T>
    Tensor<T, 2> matmul_transpose(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
        // A: (m, n), B: (p, n) -> resultado: (m, p)
        size_t m = A.shape()[0];
        size_t n = A.shape()[1];
        size_t p = B.shape()[0];
        
        Tensor<T, 2> result(m, p);
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    sum += A(i, k) * B(j, k);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }

    // Función auxiliar para multiplicación de matrices: C = A @ B
    template<typename T>
    Tensor<T, 2> matmul(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
        // A: (m, n), B: (n, p) -> resultado: (m, p)
        size_t m = A.shape()[0];
        size_t n = A.shape()[1];
        size_t p = B.shape()[1];
        
        Tensor<T, 2> result(m, p);
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }

    // Capa Dense (Fully Connected)
    template<typename T>
    class Dense final : public ILayer<T> {
    private:
        Tensor<T, 2> W_;  // Pesos: (out_features, in_features)
        Tensor<T, 2> b_;  // Sesgos: (1, out_features)
        Tensor<T, 2> X_;  // Entrada almacenada para backward
        Tensor<T, 2> dW_; // Gradiente de pesos almacenado
        Tensor<T, 2> db_; // Gradiente de sesgos almacenado
        size_t in_features_;
        size_t out_features_;

    public:
        // Constructor con funciones de inicialización
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
            : in_features_(in_f), out_features_(out_f),
              W_(out_f, in_f), b_(1, out_f),
              dW_(out_f, in_f), db_(1, out_f) {
            // Inicializar pesos y sesgos
            init_w_fun(W_);
            init_b_fun(b_);
            // Inicializar gradientes a cero
            dW_.fill(0);
            db_.fill(0);
        }

        // Forward pass: Y = X @ W^T + b
        Tensor<T, 2> forward(const Tensor<T, 2>& X) override {
            // Almacenar entrada para backward
            X_ = X;
            
            // Calcular Y = X @ W^T
            Tensor<T, 2> Y = matmul_transpose(X, W_);
            
            // Agregar sesgo a cada fila
            for (size_t i = 0; i < Y.shape()[0]; ++i) {
                for (size_t j = 0; j < Y.shape()[1]; ++j) {
                    Y(i, j) += b_(0, j);
                }
            }
            
            return Y;
        }

        // Backward pass: calcula gradientes
        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            // dZ: (batch_size, out_features)
            // Calcular dX = dZ @ W
            Tensor<T, 2> dX = matmul(dZ, W_);
            
            // Calcular dW = dZ^T @ X
            // dZ^T: (out_features, batch_size)
            // X: (batch_size, in_features)
            // dW: (out_features, in_features)
            Tensor<T, 2> dZ_T(dZ.shape()[1], dZ.shape()[0]);
            for (size_t i = 0; i < dZ.shape()[0]; ++i) {
                for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                    dZ_T(j, i) = dZ(i, j);
                }
            }
            
            dW_ = matmul(dZ_T, X_);
            
            // Calcular db = suma de dZ por columnas
            db_.fill(0);
            for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                T sum = 0;
                for (size_t i = 0; i < dZ.shape()[0]; ++i) {
                    sum += dZ(i, j);
                }
                db_(0, j) = sum;
            }
            
            return dX;
        }

        // Actualizar parámetros con un optimizador
        void update_params(IOptimizer<T>& optimizer) override {
            // Actualizar pesos y sesgos usando el optimizador
            optimizer.update(W_, dW_);
            optimizer.update(b_, db_);
        }
    };
    
    // Alias para compatibilidad con tests
    template<typename T>
    using DenseLayer = Dense<T>;

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
