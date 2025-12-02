

//

// Tensor - Proyecto Final Programación III

// Implementación organizada por preguntas

//



#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H



#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <functional>



namespace utec::algebra {



template<typename T, size_t N>

class Tensor {
private:
    std::array<size_t, N> shape_;
    std::vector<T> data_;
    std::array<size_t, N> strides_;
    // Calcula los strides para indexación row-major
    void calculate_strides() {
        if (N == 0) return;
        strides_[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
    // Calcula el tamaño total del tensor
    size_t calculate_size() const {
        size_t total = 1;
        for (size_t dim : shape_) {
            total *= dim;
        }
        return total;
    }
    // Helper para inicializar shape desde parámetros variádicos
    template<typename... Dims>
    void init_shape_from_params(Dims... dims) {
        size_t idx = 0;
        ((shape_[idx++] = static_cast<size_t>(dims)), ...);
    }
public:
    // QUESTION #1: Creación, acceso, fill e impresión
    Tensor() = default;
    // Constructor variádico
    template<typename... Dims>
    explicit Tensor(Dims... dims) {
        // Validar que el número de dimensiones coincida
        static_assert(sizeof...(dims) <= 10, "Demasiadas dimensiones");
        if (sizeof...(dims) != N) {
            throw std::invalid_argument("numero de dimensiones no coinciden con " + std::to_string(N));

        }
        // Inicializar shape
        init_shape_from_params(dims...);
        // Calcular strides
        calculate_strides();
        // Reservar memoria e inicializar con valor por defecto
        size_t total_size = calculate_size();
        data_.resize(total_size, T{});

    }
    // Método shape() - retorna las dimensiones del tensor
    std::array<size_t, N> shape() const {
        return shape_;
    }
    // Método size() - retorna el número total de elementos
    size_t size() const {
        return data_.size();
    }
    // Método fill() - llena el tensor con un valor
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    // Operador de asignación con initializer_list
    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("El tamaño de los datos no coinciden con el tamaño del tensor");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    // Iteradores para compatibilidad con STL
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }
    // Acceso a miembros internos (para funciones amigas)
    const std::vector<T>& get_data() const { return data_; }
    const std::array<size_t, N>& get_strides() const { return strides_; }
    // QUESTION #2: Reshape
    // Método reshape() - cambia las dimensiones del tensor
    template<typename... Dims>
    void reshape(Dims... dims) {
        // Validar que el número de dimensiones coincida
        if (sizeof...(dims) != N) {
            throw std::invalid_argument("el numero de dimensiones no coinciden con" + std::to_string(N));
        }

        // Calcular nuevo shape y tamaño
        std::array<size_t, N> new_shape;
        size_t idx = 0;
        ((new_shape[idx++] = static_cast<size_t>(dims)), ...);
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        // Ajustar el tamaño del vector de datos
        if (new_size > data_.size()) {
            // Expandir: rellenar con valor por defecto
            data_.resize(new_size, T{});
        } else if (new_size < data_.size()) {
            // Reducir: truncar datos
            data_.resize(new_size);
        }
        // Si new_size == data_.size(), no hay cambio en el tamaño
        // Actualizar shape y recalcular strides
        shape_ = new_shape;
        calculate_strides();
    }




    // QUESTION #3: Suma y resta de tensores (2 points)
    // Operador () para acceso por índices - versión no const
    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(indices) == N, "El numero de indices debe coincidir con las dimensiones del tensor");
        // Calcular índice lineal usando strides
        size_t linear_idx = 0;
        size_t dim_idx = 0;
        ((linear_idx += static_cast<size_t>(indices) * strides_[dim_idx++]), ...);
        return data_[linear_idx];
    }
    // Operador () para acceso por índices - versión const
    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(indices) == N, "El numero de indices debe coincidir con las dimensiones del tensor");
        // Calcular índice lineal usando strides
        size_t linear_idx = 0;
        size_t dim_idx = 0;
        ((linear_idx += static_cast<size_t>(indices) * strides_[dim_idx++]), ...);
        return data_[linear_idx];
    }
    // Validar compatibilidad de shapes (exacta, sin broadcasting)
    bool shapes_match(const Tensor<T, N>& other) const {
        return shape_ == other.shape_;
    }
    // Validar compatibilidad de broadcasting según reglas de NumPy
    bool is_broadcastable(const Tensor<T, N>& other) const {
        for (size_t i = 0; i < N; ++i) {
            if (shape_[i] != other.shape_[i] && shape_[i] != 1 && other.shape_[i] != 1) {
                return false;
            }
        }
        return true;
    }

    // Calcular shape resultante después de broadcasting
    std::array<size_t, N> broadcast_shape(const Tensor<T, N>& other) const {
        std::array<size_t, N> result_shape;
        for (size_t i = 0; i < N; ++i) {
            result_shape[i] = std::max(shape_[i], other.shape_[i]);
        }
        return result_shape;
    }

    // Convertir índice multi-dimensional a índice lineal con broadcasting

    size_t broadcast_index(const std::array<size_t, N>& indices) const {
        size_t linear_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            // Si la dimensión es 1, usar índice 0 (broadcasting)
            size_t idx = (shape_[i] == 1) ? 0 : indices[i];
            linear_idx += idx * strides_[i];
        }
        return linear_idx;
    }

    // Operador + (suma de tensores con broadcasting)
    Tensor<T, N> operator+(const Tensor<T, N>& other) const {
        // Si shapes coinciden exactamente, usar operación rápida
        if (shapes_match(other)) {
            Tensor<T, N> result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] += other.data_[i];
            }
            return result;
        }

        // Verificar si es compatible con broadcasting

        if (!is_broadcastable(other)) {
        }
        // Aplicar broadcasting
        auto result_shape = broadcast_shape(other);
        Tensor<T, N> result;
        result.shape_ = result_shape;
        result.calculate_strides();
        result.data_.resize(result.calculate_size());
        // Iterar sobre todos los índices del resultado
        std::array<size_t, N> indices{};
        for (size_t i = 0; i < result.data_.size(); ++i) {
            // Convertir índice lineal a multi-dimensional
            size_t temp = i;
            for (int d = N - 1; d >= 0; --d) {
                indices[d] = temp % result.shape_[d];
                temp /= result.shape_[d];
            }
            // Obtener valores con broadcasting
            size_t idx1 = broadcast_index(indices);
            size_t idx2 = other.broadcast_index(indices);
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }
        return result;
    }
    // Operador - (resta de tensores con broadcasting)
    Tensor<T, N> operator-(const Tensor<T, N>& other) const {
        // Si shapes coinciden exactamente, usar operación rápida
        if (shapes_match(other)) {
            Tensor<T, N> result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] -= other.data_[i];
            }
            return result;
        }
        // Verificar si es compatible con broadcasting
        if (!is_broadcastable(other)) {

        }

        // Aplicar broadcasting
        auto result_shape = broadcast_shape(other);
        Tensor<T, N> result;
        result.shape_ = result_shape;
        result.calculate_strides();
        result.data_.resize(result.calculate_size());
        // Iterar sobre todos los índices del resultado
        std::array<size_t, N> indices{};
        for (size_t i = 0; i < result.data_.size(); ++i) {
            // Convertir índice lineal a multi-dimensional
            size_t temp = i;
            for (int d = N - 1; d >= 0; --d) {
                indices[d] = temp % result.shape_[d];
                temp /= result.shape_[d];
            }

            // Obtener valores con broadcasting
            size_t idx1 = broadcast_index(indices);
            size_t idx2 = other.broadcast_index(indices);
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }
        return result;
    }
    // Operador += (suma in-place con escalar)
    Tensor<T, N>& operator+=(const T& scalar) {
        for (auto& val : data_) {
            val += scalar;
        }
        return *this;
    }


    // QUESTION #4: Multiplicación y Operaciones con escalares (2 points)
    // Operador * (multiplicación element-wise de tensores con broadcasting)

    Tensor<T, N> operator*(const Tensor<T, N>& other) const {

        // Si shapes coinciden exactamente, usar operación rápida

        if (shapes_match(other)) {

            Tensor<T, N> result = *this;

            for (size_t i = 0; i < data_.size(); ++i) {

                result.data_[i] *= other.data_[i];
            }
            return result;
        }
        // Verificar si es compatible con broadcasting
        if (!is_broadcastable(other)) {
        }
        // Aplicar broadcasting

        auto result_shape = broadcast_shape(other);

        Tensor<T, N> result;

        result.shape_ = result_shape;

        result.calculate_strides();

        result.data_.resize(result.calculate_size());

        

        // Iterar sobre todos los índices del resultado

        std::array<size_t, N> indices{};

        for (size_t i = 0; i < result.data_.size(); ++i) {

            // Convertir índice lineal a multi-dimensional

            size_t temp = i;

            for (int d = N - 1; d >= 0; --d) {

                indices[d] = temp % result.shape_[d];

                temp /= result.shape_[d];

            }

            

            // Obtener valores con broadcasting

            size_t idx1 = broadcast_index(indices);

            size_t idx2 = other.broadcast_index(indices);

            result.data_[i] = data_[idx1] * other.data_[idx2];

        }

        

        return result;

    }

    

    // Operador + con escalar (tensor + scalar)

    Tensor<T, N> operator+(const T& scalar) const {

        Tensor<T, N> result = *this;

        for (auto& val : result.data_) {

            val += scalar;

        }

        return result;

    }

    

    // Operador - con escalar (tensor - scalar)

    Tensor<T, N> operator-(const T& scalar) const {

        Tensor<T, N> result = *this;

        for (auto& val : result.data_) {

            val -= scalar;

        }

        return result;

    }

    

    // Operador * con escalar (tensor * scalar)

    Tensor<T, N> operator*(const T& scalar) const {

        Tensor<T, N> result = *this;

        for (auto& val : result.data_) {

            val *= scalar;

        }

        return result;

    }

    

    // Operador / con escalar (tensor / scalar)

    Tensor<T, N> operator/(const T& scalar) const {

        Tensor<T, N> result = *this;

        for (auto& val : result.data_) {

            val /= scalar;

        }

        return result;

    }
    // OPERADOR DE IMPRESIÓN
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& tensor) {

        if constexpr (N == 1) {

            // Tensor 1D: a b c ...

            for (size_t i = 0; i < tensor.data_.size(); ++i) {

                if (i > 0) os << " ";

                os << tensor.data_[i];

            }

        } else if constexpr (N == 2) {

            // Tensor 2D: matriz

            os << "{\n";

            for (size_t i = 0; i < tensor.shape_[0]; ++i) {

                for (size_t j = 0; j < tensor.shape_[1]; ++j) {

                    if (j > 0) os << " ";

                    os << tensor.data_[i * tensor.shape_[1] + j];

                }

                os << "\n";

            }

            os << "}";

        } else if constexpr (N == 3) {

            // Tensor 3D

            os << "{\n";

            for (size_t i = 0; i < tensor.shape_[0]; ++i) {

                os << "{\n";

                for (size_t j = 0; j < tensor.shape_[1]; ++j) {

                    for (size_t k = 0; k < tensor.shape_[2]; ++k) {

                        if (k > 0) os << " ";

                        size_t idx = i * tensor.strides_[0] + j * tensor.strides_[1] + k;

                        os << tensor.data_[idx];

                    }

                    os << "\n";

                }

                os << "}\n";

            }

            os << "}";

        } else {

            // Para N > 3, impresión genérica simplificada

            os << "[Tensor with shape: ";

            for (size_t i = 0; i < N; ++i) {

                if (i > 0) os << "x";

                os << tensor.shape_[i];

            }

            os << "]";

        }

        return os;

    }

};





// Operador + con escalar a la izquierda (scalar + tensor)
template<typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor + scalar;
}
// Operadr * con escalar a la izquierda (scalar * tensor)
template<typename T, size_t N>
Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor * scalar;
}




// QUESTION #7: Transpose 2D (2 points)
// Función transpose_2d - transpone las últimas 2 dimensiones
template<typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
    // Validar que tenga al menos 2 dimensiones
    if constexpr (N < 2) {
    }
    // Obtener shape original
    auto original_shape = tensor.shape();
    // Crear nuevo shape con las últimas 2 dimensiones intercambiadas
    std::array<size_t, N> new_shape = original_shape;
    std::swap(new_shape[N-2], new_shape[N-1]);
    // Crear tensor resultado usando constructor variádico
    // Necesitamos construir el tensor con el nuevo shape
    Tensor<T, N> result;
    // Calcular el tamaño total
    size_t total_size = 1;
    for (size_t dim : new_shape) {
        total_size *= dim;
    }
    // Crear vector temporal para los datos
    std::vector<T> new_data(total_size);
    // Obtener strides originales
    auto orig_strides = tensor.get_strides();
    // Calcular strides para el resultado
    std::array<size_t, N> new_strides;
    new_strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }
    // Copiar datos con transposición
    std::array<size_t, N> indices{};
    // Función auxiliar para iterar sobre todos los índices
    std::function<void(size_t)> iterate = [&](size_t dim) {
        if (dim == N) {
            // Calcular índice en tensor original
            size_t src_idx = 0;
            for (size_t d = 0; d < N; ++d) {
                src_idx += indices[d] * orig_strides[d];

            }
            // Calcular índice en tensor resultado (con últimas 2 dims intercambiadas)
            std::array<size_t, N> transposed_indices = indices;
            std::swap(transposed_indices[N-2], transposed_indices[N-1]);
            size_t dst_idx = 0;
            for (size_t d = 0; d < N; ++d) {
                dst_idx += transposed_indices[d] * new_strides[d];

            }
            new_data[dst_idx] = tensor.get_data()[src_idx];
            return;
        }
        for (size_t i = 0; i < original_shape[dim]; ++i) {

            indices[dim] = i;

            iterate(dim + 1);

        }

    };
    iterate(0);

    

    // Construir el resultado usando reshape

    // Primero crear un tensor con el tamaño correcto

    if constexpr (N == 2) {

        result = Tensor<T, N>(new_shape[0], new_shape[1]);

    } else if constexpr (N == 3) {

        result = Tensor<T, N>(new_shape[0], new_shape[1], new_shape[2]);

    } else if constexpr (N == 4) {

        result = Tensor<T, N>(new_shape[0], new_shape[1], new_shape[2], new_shape[3]);

    }

    

    // Copiar los datos

    std::copy(new_data.begin(), new_data.end(), result.begin());

    

    return result;

}




// QUESTION #7: Multiplicación de matrices

// Función matrix_product - multiplicación matricial de las últimas 2 dimensiones
template<typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& a, const Tensor<T, N>& b) {
    static_assert(N >= 2);
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    // Extraer dimensiones de las matrices (últimas 2 dimensiones)
    size_t M = shape_a[N-2];  // Filas de A
    size_t K = shape_a[N-1];  // Columnas de A / Filas de B
    size_t P = shape_b[N-1];  // Columnas de B
    // Validar que las dimensiones de matriz sean compatibles
    if (shape_b[N-2] != K) {
    }
    // Validar que las dimensiones batch coincidan (si N > 2)
    if constexpr (N > 2) {
        for (size_t i = 0; i < N-2; ++i) {
            if (shape_a[i] != shape_b[i]) {
            }
        }
    }
    // Crear shape del resultado
    std::array<size_t, N> result_shape = shape_a;
    result_shape[N-1] = P;  // Última dimensión es P (columnas de B)
    // Crear tensor resultado
    Tensor<T, N> result;
    if constexpr (N == 2) {
        result = Tensor<T, N>(M, P);
    } else if constexpr (N == 3) {
        result = Tensor<T, N>(result_shape[0], M, P);
    } else if constexpr (N == 4) {
        result = Tensor<T, N>(result_shape[0], result_shape[1], M, P);
    }
    // Inicializar resultado con ceros
    std::fill(result.begin(), result.end(), T{0});
    // Obtener strides
    auto strides_a = a.get_strides();
    auto strides_b = b.get_strides();
    auto strides_r = result.get_strides();
    // Calcular número de matrices (producto de dimensiones batch)
    size_t num_matrices = 1;
    if constexpr (N > 2) {
        for (size_t i = 0; i < N-2; ++i) {
            num_matrices *= shape_a[i];
        }
    }
    // Iterar sobre cada matriz en el batch
    for (size_t batch = 0; batch < num_matrices; ++batch) {
        size_t batch_offset_a = 0;
        size_t batch_offset_b = 0;
        size_t batch_offset_r = 0;
        if constexpr (N > 2) {
            size_t temp = batch;
            for (int d = N-3; d >= 0; --d) {
                size_t idx = temp % shape_a[d];
                temp /= shape_a[d];
                batch_offset_a += idx * strides_a[d];
                batch_offset_b += idx * strides_b[d];
                batch_offset_r += idx * strides_r[d];
            }
        }
        // Multiplicación matricial: C[i,j] = sum(A[i,k] * B[k,j])
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                T sum = T{0};
                for (size_t k = 0; k < K; ++k) {
                    size_t idx_a = batch_offset_a + i * strides_a[N-2] + k * strides_a[N-1];
                    size_t idx_b = batch_offset_b + k * strides_b[N-2] + j * strides_b[N-1];
                    sum += a.get_data()[idx_a] * b.get_data()[idx_b];
                }
                size_t idx_r = batch_offset_r + i * strides_r[N-2] + j * strides_r[N-1];
                result.begin()[idx_r] = sum;
            }
        }
    }
    return result;

}



} // namespace utec::algebra



#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
