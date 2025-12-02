/**
 * @file ControllerDemo.h
 * @brief Aplicación: Control simplificado de un sistema
 * @details Recibe como entrada un estado reducido de un sistema (posición, velocidad)
 *          y decide una acción apropiada (mover izquierda/derecha, acelerar/detener)
 */

#ifndef UTEC_APPS_CONTROLLER_DEMO_H
#define UTEC_APPS_CONTROLLER_DEMO_H

#include "../nn/neural_network.h"
#include <vector>
#include <string>

namespace utec::apps {

/**
 * @class ControllerDemo
 * @brief Controlador de sistema usando red neuronal
 */
class ControllerDemo {
public:
    enum class Action {
        MOVE_LEFT = 0,
        STAY = 1,
        MOVE_RIGHT = 2,
        ACCELERATE = 3,
        DECELERATE = 4
    };
    
    struct SystemState {
        float position;      // Posición actual [-1.0, 1.0]
        float velocity;      // Velocidad [-1.0, 1.0]
        float target_pos;    // Posición objetivo
        float time_step;     // Paso de tiempo normalizado
    };
    
    ControllerDemo();
    
    /**
     * @brief Genera datos de entrenamiento simulando un entorno
     * @param num_episodes Número de episodios de simulación
     * @param steps_per_episode Pasos por episodio
     * @return Par (states, actions) para entrenamiento
     */
    std::pair<nn::Tensor2D, nn::Tensor2D> generate_training_data(
        size_t num_episodes, size_t steps_per_episode);
    
    /**
     * @brief Entrena el controlador
     * @param epochs Número de épocas
     * @param batch_size Tamaño del lote
     * @param learning_rate Tasa de aprendizaje
     */
    void train(size_t epochs, size_t batch_size = 32, float learning_rate = 0.01f);
    
    /**
     * @brief Decide una acción basada en el estado
     * @param state Estado actual del sistema
     * @return Acción recomendada
     */
    Action decide_action(const SystemState& state);
    
    /**
     * @brief Obtiene la confianza de la acción decidida
     * @param state Estado actual del sistema
     * @return Confianza (0.0 a 1.0)
     */
    float get_action_confidence(const SystemState& state);
    
    /**
     * @brief Simula el sistema con el controlador entrenado
     * @param num_steps Número de pasos de simulación
     * @param target_position Posición objetivo
     * @return Historial de posiciones y acciones
     */
    struct SimulationResult {
        std::vector<float> positions;
        std::vector<float> velocities;
        std::vector<int> actions;
        float final_error;
    };
    
    SimulationResult simulate(size_t num_steps, float target_position);
    
    /**
     * @brief Evalúa el controlador en múltiples simulaciones
     * @param num_simulations Número de simulaciones
     * @param num_steps Pasos por simulación
     * @return Error promedio
     */
    float evaluate(size_t num_simulations, size_t num_steps);
    
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
    static constexpr size_t STATE_FEATURES = 4;    // [position, velocity, target_pos, time]
    static constexpr size_t NUM_ACTIONS = 5;       // [left, stay, right, accelerate, decelerate]
    
    nn::Tensor2D state_to_tensor(const SystemState& state);
    SystemState apply_action(const SystemState& state, Action action);
};

} // namespace utec::apps

#endif // UTEC_APPS_CONTROLLER_DEMO_H
