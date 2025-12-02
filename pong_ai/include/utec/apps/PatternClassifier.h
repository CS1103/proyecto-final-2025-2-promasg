/**
 * @file PatternClassifier.h
 * @brief Aplicación: Clasificación de patrones geométricos
 * @details Reconoce formas básicas (círculos, cuadrados, triángulos) 
 *          en entradas vectoriales simuladas
 */

#ifndef UTEC_APPS_PATTERN_CLASSIFIER_H
#define UTEC_APPS_PATTERN_CLASSIFIER_H

#include "../nn/neural_network.h"
#include <vector>
#include <string>

namespace utec::apps {

/**
 * @class PatternClassifier
 * @brief Clasificador de patrones geométricos usando red neuronal
 */
class PatternClassifier {
public:
    enum class Pattern {
        CIRCLE = 0,
        SQUARE = 1,
        TRIANGLE = 2
    };
    
    PatternClassifier();
    
    /**
     * @brief Genera datos de entrenamiento sintéticos
     * @param num_samples Número de muestras a generar
     * @return Par (inputs, targets) para entrenamiento
     */
    std::pair<nn::Tensor2D, nn::Tensor2D> generate_training_data(size_t num_samples);
    
    /**
     * @brief Entrena el clasificador
     * @param epochs Número de épocas
     * @param batch_size Tamaño del lote
     * @param learning_rate Tasa de aprendizaje
     */
    void train(size_t epochs, size_t batch_size = 32, float learning_rate = 0.01f);
    
    /**
     * @brief Clasifica un patrón
     * @param input Vector de características [1, input_features]
     * @return Patrón clasificado
     */
    Pattern classify(const nn::Tensor2D& input);
    
    /**
     * @brief Obtiene la confianza de la predicción
     * @param input Vector de características
     * @return Confianza (0.0 a 1.0)
     */
    float get_confidence(const nn::Tensor2D& input);
    
    /**
     * @brief Evalúa el modelo en datos de prueba
     * @param test_inputs Datos de prueba
     * @param test_targets Etiquetas de prueba
     * @return Precisión (0.0 a 1.0)
     */
    float evaluate(const nn::Tensor2D& test_inputs, const nn::Tensor2D& test_targets);
    
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
    static constexpr size_t INPUT_FEATURES = 4;   // [area, perimeter, compactness, aspect_ratio]
    static constexpr size_t OUTPUT_CLASSES = 3;   // [circle, square, triangle]
    
    nn::Tensor2D generate_circle_sample() const;
    nn::Tensor2D generate_square_sample() const;
    nn::Tensor2D generate_triangle_sample() const;
};

} // namespace utec::apps

#endif // UTEC_APPS_PATTERN_CLASSIFIER_H
