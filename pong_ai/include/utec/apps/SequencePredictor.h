/**
 * @file SequencePredictor.h
 * @brief Aplicación: Predicción de series numéricas
 * @details Entrena la red con datos sintéticos de una secuencia numérica
 *          y evalúa su capacidad de predecir el siguiente valor
 */

#ifndef UTEC_APPS_SEQUENCE_PREDICTOR_H
#define UTEC_APPS_SEQUENCE_PREDICTOR_H

#include "../nn/neural_network.h"
#include <vector>
#include <string>

namespace utec::apps {

/**
 * @class SequencePredictor
 * @brief Predictor de series numéricas usando red neuronal
 */
class SequencePredictor {
public:
    SequencePredictor();
    
    /**
     * @brief Genera una secuencia sintética
     * @param length Longitud de la secuencia
     * @param sequence_type Tipo de secuencia (0=lineal, 1=cuadrática, 2=senoidal)
     * @return Vector de valores
     */
    std::vector<float> generate_sequence(size_t length, int sequence_type = 0);
    
    /**
     * @brief Prepara datos de entrenamiento a partir de una secuencia
     * @param sequence Secuencia de entrada
     * @param window_size Tamaño de la ventana (número de pasos anteriores)
     * @return Par (inputs, targets) para entrenamiento
     */
    std::pair<nn::Tensor2D, nn::Tensor2D> prepare_training_data(
        const std::vector<float>& sequence, size_t window_size = 5);
    
    /**
     * @brief Entrena el predictor
     * @param sequence Secuencia de entrenamiento
     * @param epochs Número de épocas
     * @param window_size Tamaño de la ventana
     * @param learning_rate Tasa de aprendizaje
     */
    void train(const std::vector<float>& sequence, size_t epochs, 
               size_t window_size = 5, float learning_rate = 0.01f);
    
    /**
     * @brief Predice el siguiente valor en la secuencia
     * @param history Últimos valores de la secuencia [1, window_size]
     * @return Predicción del siguiente valor
     */
    float predict_next(const nn::Tensor2D& history);
    
    /**
     * @brief Predice múltiples pasos adelante
     * @param history Últimos valores de la secuencia
     * @param steps Número de pasos a predecir
     * @return Vector de predicciones
     */
    std::vector<float> predict_ahead(const nn::Tensor2D& history, size_t steps);
    
    /**
     * @brief Evalúa el predictor en datos de prueba
     * @param test_sequence Secuencia de prueba
     * @param window_size Tamaño de la ventana
     * @return Error cuadrático medio (MSE)
     */
    float evaluate(const std::vector<float>& test_sequence, size_t window_size = 5);
    
    /**
     * @brief Guarda el modelo entrenado
     * @param filename Ruta del archivo
     */
    void save_model(const std::string& filename);
    
    /**
     * @brief Carga un modelo entrenado
     * @param filename Ruta del archivo
     */
    void load_model(const std::string& filename);
    
    /**
     * @brief Obtiene la red neuronal
     */
    utec::NeuralNetwork& get_network() { return network_; }
    const utec::NeuralNetwork& get_network() const { return network_; }
    
private:
    utec::NeuralNetwork network_;
    static constexpr size_t HIDDEN_UNITS = 32;
};

} // namespace utec::apps

#endif // UTEC_APPS_SEQUENCE_PREDICTOR_H
