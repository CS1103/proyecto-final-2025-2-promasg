[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)

# Proyecto Final 2025-2: Neural Network

**CS2013 Programación III** · Proyecto Final 2025-2

Una implementación completa de red neuronal desde cero en C++20 moderno, incluyendo álgebra genérica, framework de aprendizaje profundo y aplicaciones prácticas.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Datos Generales](#datos-generales)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Investigación Teórica](#investigación-teórica)
- [Diseño e Implementación](#diseño-e-implementación)
- [Manual de Uso y Casos de Prueba](#manual-de-uso-y-casos-de-prueba)
- [Ejecución](#ejecución)
- [Análisis del Rendimiento](#análisis-del-rendimiento)
- [Trabajo en Equipo](#trabajo-en-equipo)
- [Conclusiones](#conclusiones)
- [Bibliografía](#bibliografía)
- [Licencia](#licencia)

---

## Descripción

Implementación completa de una red neuronal multicapa en C++20, incluyendo álgebra genérica de tensores, framework de aprendizaje profundo con múltiples funciones de activación y optimizadores, y tres aplicaciones prácticas: clasificación de patrones geométricos, predicción de series numéricas y control de sistemas.

## Características

### Epic 1: Álgebra Genérica
- **Tensor<T, Rank>**: Tensores multidimensionales genéricos
- **Acceso variádico**: `tensor(i, j, k)` para cualquier dimensión
- **Operaciones**: Suma, resta, multiplicación escalar, división
- **Almacenamiento eficiente**: Contiguo en memoria

### Epic 2: Red Neuronal Full-Stack
- **Capas densas** (fully connected)
- **Funciones de activación**: ReLU, Sigmoid, Tanh, Linear, Softmax
- **Funciones de pérdida**: MSE, CrossEntropy, BinaryCrossEntropy, MAE
- **Optimizadores**: SGD, Adam, RMSprop
- **Forward/Backward pass** completo
- **Serialización** de modelos (interfaz preparada, implementación futura)

### Epic 3: Aplicaciones Prácticas
- **PatternClassifier**: Clasificación de patrones geométricos
- **SequencePredictor**: Predicción de series numéricas
- **ControllerDemo**: Control de sistema

---

## Datos Generales

- **Tema**: Neural Network
- **Curso**: CS2013 Programación III
- **Semestre**: 2025-2
- **Autor**: Marco Sabino Guardian

---

## Requisitos e Instalación

### Requisitos
- **Compilador**: GCC 11+ o Clang 12+ (C++20)
- **CMake**: 3.18+
- **Sistema**: Linux, macOS o Windows (WSL)
- **Sin dependencias externas** (implementación desde cero)

### Compilar

```bash
cd pong_ai
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Ejecutar Pruebas

```bash
cd build
make test
```

O ejecutar tests individuales desde el directorio `build`:
```bash
./test_tensor
./test_neural_network
./test_applications
```

### Ejecutar Demostración

```bash
cd build
./pong_ai_demo
```

### Ejecutar Ejemplo de Uso

```bash
cd build
./example_usage
```

---

## Estructura del Proyecto

```
pong_ai/
├── include/utec/
│   ├── algebra/
│   │   └── Tensor.h                    # Tensor genérico (header-only)
│   ├── nn/
│   │   ├── nn_interfaces.h             # Interfaces base
│   │   ├── nn_dense.h                  # Capas densas
│   │   ├── nn_activation.h             # Funciones de activación
│   │   ├── nn_loss.h                   # Funciones de pérdida
│   │   ├── nn_optimizer.h              # Optimizadores
│   │   └── neural_network.h            # Red neuronal completa
│   └── apps/
│       ├── PatternClassifier.h
│       ├── SequencePredictor.h
│       └── ControllerDemo.h
├── src/utec/apps/
│   ├── PatternClassifier.cpp
│   ├── SequencePredictor.cpp
│   ├── ControllerDemo.cpp
│   └── main.cpp                        # Demostración
├── tests/
│   ├── test_tensor.cpp
│   ├── test_neural_network.cpp
│   └── test_applications.cpp
├── examples/
│   └── example_usage.cpp
├── CMakeLists.txt
└── README.md
```

---

## Investigación Teórica

**Objetivo**: Implementar una red neuronal multicapa con soporte para múltiples funciones de activación y optimizadores.

**Contenido implementado**:

1. **Álgebra Genérica**: Tensor<T, Rank> con acceso variádico y broadcasting
2. **Capas Neuronales**: Dense (fully connected) con forward/backward pass
3. **Funciones de Activación**: ReLU, Sigmoid, Tanh, Linear, Softmax
4. **Optimizadores**: SGD, Adam, RMSprop con momentum y adaptive learning rates
5. **Funciones de Pérdida**: MSE, CrossEntropy, BinaryCrossEntropy, MAE
6. **Algoritmos**: Backpropagation, Gradient Descent, Batch Training

---

## Diseño e Implementación

### Arquitectura de la Solución

**Patrones de diseño**:
- **Strategy**: Interfaces intercambiables para activaciones, pérdida y optimizadores
- **Factory**: Creación de capas con activaciones específicas
- **Template**: Tensor genérico para cualquier tipo y dimensión

### Casos de Prueba

- Test unitario de Tensor (creación, acceso, operaciones)
- Test de capas densas (forward/backward pass)
- Test de funciones de activación (ReLU, Sigmoid, Tanh, Linear, Softmax)
- Test de optimizadores (SGD, Adam, RMSprop)
- Test de aplicaciones (PatternClassifier, SequencePredictor, ControllerDemo)

---

## Manual de Uso y Casos de Prueba

### Cómo Ejecutar

#### Demo Principal
```bash
cd pong_ai/build
./pong_ai_demo
```

#### Ejemplo de Uso Básico
```bash
cd pong_ai/build
./example_usage
```

#### Tests Unitarios
```bash
cd pong_ai/build

# Test de Tensor
./test_tensor

# Test de Red Neuronal
./test_neural_network

# Test de Aplicaciones
./test_applications

# Todos los tests
make test
```

### Casos de Prueba Implementados

#### 1. Test Unitario de Capa Densa
- **Ubicación**: `tests/test_neural_network.cpp::test_dense_layer()`
- **Verifica**: Creación de capa, forward pass, dimensiones de salida
- **Ejecutar**: `./test_neural_network`

#### 2. Test de Función de Activación ReLU
- **Ubicación**: `tests/test_neural_network.cpp::test_activation_functions()`
- **Verifica**: 
  - ReLU: `max(0, x)` correcto
  - Sigmoid: valores entre 0 y 1
  - Linear: función identidad
- **Ejecutar**: `./test_neural_network`

#### 3. Test de Convergencia en Dataset de Ejemplo
- **Ubicación**: `tests/test_neural_network.cpp::test_training()`
- **Verifica**: 
  - Reducción de pérdida durante entrenamiento
  - Datos XOR-like (4 muestras, 2 características)
  - Optimizador Adam con learning rate 0.01
- **Resultado esperado**: Pérdida inicial ~0.24, final ~0.23
- **Ejecutar**: `./test_neural_network`

#### 4. Tests de Aplicaciones Prácticas
- **PatternClassifier**: Clasificación con precisión ~77-84%
- **SequencePredictor**: Predicción de series numéricas
- **ControllerDemo**: Control de sistema con error < 0.1
- **Ejecutar**: `./test_applications`

### Personalización

Los tests pueden personalizarse modificando:
- **Parámetros de red**: Tamaño de capas, funciones de activación
- **Datos de entrenamiento**: Número de muestras, características
- **Hiperparámetros**: Learning rate, épocas, batch size

Ejemplo de personalización en `examples/example_usage.cpp`:
```cpp
// Cambiar arquitectura
nn.add_dense_layer(4, 16, std::make_shared<ReLU<float>>());  // Más neuronas
nn.add_dense_layer(16, 2, std::make_shared<Softmax<float>>());

// Cambiar optimizador
RMSprop<float> optimizer(0.001f);  // En lugar de Adam
```

---

## Ejecución

### Pasos para Ejecutar

1. **Compilar el proyecto**:
   ```bash
   cd pong_ai
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

2. **Ejecutar pruebas unitarias**:
   ```bash
   make test
   ```
   
   O ejecutar tests individuales:
   ```bash
   ./test_tensor
   ./test_neural_network
   ./test_applications
   ```

3. **Ejecutar demostración completa**:
   ```bash
   ./pong_ai_demo
   ```
   
   La demo ejecuta automáticamente las 3 aplicaciones:
   - PatternClassifier: Entrena y clasifica patrones geométricos
   - SequencePredictor: Predice valores en series numéricas
   - ControllerDemo: Simula control de sistema

4. **Ejecutar ejemplo básico**:
   ```bash
   ./example_usage
   ```

### Demo de Ejemplo

La demostración completa (`pong_ai_demo`) muestra:

1. **Pattern Classifier**:
   - Entrenamiento: 100 épocas, batch size 32
   - Precisión típica: 77-84% en datos de prueba
   - Clasifica: Círculos, Cuadrados, Triángulos

2. **Sequence Predictor**:
   - Entrenamiento: 100 épocas, ventana de 5 pasos
   - Predice: Próximos valores en secuencias lineales/cuadráticas/senoidales
   - MSE típico: 13-24 en secuencias de prueba

3. **Controller Demo**:
   - Entrenamiento: 100 épocas, 20 episodios
   - Simulación: 15 pasos hacia posición objetivo
   - Error final típico: < 0.1

### Preparación de Datos

**Nota**: Las aplicaciones actuales generan datos sintéticos automáticamente. No se requiere preparación manual de datos.

- **PatternClassifier**: Genera 300 muestras de entrenamiento (100 por clase)
- **SequencePredictor**: Genera secuencias de 50-100 valores
- **ControllerDemo**: Genera 200 muestras de simulación (20 episodios × 10 pasos)

### Evaluación de Resultados

Cada aplicación incluye métodos de evaluación:

```cpp
// PatternClassifier
float accuracy = classifier.evaluate(test_inputs, test_targets);

// SequencePredictor  
float mse = predictor.evaluate(test_sequence, window_size);

// ControllerDemo
float avg_error = controller.evaluate(num_simulations, num_steps);
```

### Aplicaciones Incluidas

- **PatternClassifier**: Clasifica patrones geométricos (círculos, cuadrados, triángulos)
- **SequencePredictor**: Predice el siguiente valor en series numéricas
- **ControllerDemo**: Controla un sistema hacia un objetivo

### Ejemplo de Uso

```cpp
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"

using namespace utec::nn;

int main() {
    // Crear red
    utec::NeuralNetwork nn;
    nn.add_dense_layer(4, 8, std::make_shared<ReLU<float>>());
    nn.add_dense_layer(8, 2, std::make_shared<Softmax<float>>());
    
    // Datos
    Tensor2D input(2, 4);
    Tensor2D targets(2, 2);
    
    // Llenar datos (ejemplo)
    // ... inicializar input y targets ...
    
    // Entrenar
    CrossEntropyLoss<float> loss_obj(input, targets);
    Adam<float> optimizer(0.001f);
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        nn.train_step(input, targets, loss_obj, optimizer);
    }
    
    // Realizar predicciones
    Tensor2D predictions = nn.predict(input);
    
    // Nota: La serialización (save/load) está en desarrollo
    // nn.save("model.bin");  // Futuro
    
    return 0;
}
```

**Para ejecutar este ejemplo:**
```bash
cd pong_ai/build
./example_usage
```

---

## Análisis del Rendimiento

### Métricas de Ejemplo (Resultados Reales)

#### PatternClassifier
- **Épocas**: 100
- **Tiempo de entrenamiento**: ~2-3 segundos (300 muestras)
- **Precisión final**: 77-84% en datos de prueba
- **Reducción de pérdida**: ~55% (de 1.0 a 0.45)

#### SequencePredictor
- **Épocas**: 100
- **Tiempo de entrenamiento**: ~1-2 segundos (45 muestras)
- **MSE final**: 13-24 en secuencias de prueba
- **Reducción de pérdida**: ~7% (de 14.3 a 13.3)

#### ControllerDemo
- **Épocas**: 100
- **Tiempo de entrenamiento**: ~2-3 segundos (200 muestras)
- **Error promedio**: < 0.1 en simulaciones
- **Reducción de pérdida**: ~51% (de 1.24 a 0.60)

#### Test de Entrenamiento (XOR-like)
- **Épocas**: 10 pasos de entrenamiento
- **Tiempo**: < 0.1 segundos
- **Pérdida inicial**: ~0.242
- **Pérdida final**: ~0.233
- **Reducción**: ~3.7%

### Complejidad de Algoritmos

#### Operaciones de Tensor

| Operación | Complejidad | Notas |
|-----------|------------|-------|
| Acceso `tensor(i,j)` | **O(1)** | Acceso directo con strides |
| Suma/Resta | **O(n·m)** | n=filas, m=columnas |
| Multiplicación elemento-wise | **O(n·m)** | Elemento a elemento |
| Multiplicación matricial | **O(n·m·k)** | n×m * m×k = n×k |
| Broadcasting | **O(max_size)** | Copia con broadcasting |
| Reshape | **O(1)** | Solo actualiza metadatos |
| Transpose | **O(n·m)** | Reorganiza datos |

### Ventajas y Desventajas

#### Ventajas
- **Código ligero**: Sin dependencias externas, solo C++20 estándar
- **Implementación educativa**: Desde cero, fácil de entender y modificar
- **Genérico y extensible**: Templates permiten cualquier tipo de dato
- **Modular**: Arquitectura clara con interfaces intercambiables
- **Tests completos**: Cobertura de todos los componentes principales

#### Desventajas
- **Sin paralelización**: Rendimiento limitado en grandes volúmenes de datos
- **Sin optimizaciones BLAS**: Multiplicaciones matriciales no optimizadas
- **Memoria secuencial**: No hay gestión avanzada de memoria

### Mejoras Futuras

#### 1. Uso de BLAS para Multiplicaciones
**Justificación**: Las multiplicaciones matriciales (`O(n·m·k)`) son el cuello de botella principal. BLAS (Basic Linear Algebra Subprograms) puede acelerar estas operaciones 10-100x usando optimizaciones específicas de CPU.

**Implementación sugerida**:
```cpp
// Reemplazar matmul() con llamadas a cblas_sgemm()
#include <cblas.h>
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1.0, A, k, B, n, 0.0, C, n);
```

#### 2. Paralelizar Entrenamiento por Lotes
**Justificación**: El procesamiento de batches es independiente y puede paralelizarse fácilmente. Con OpenMP, se puede lograr speedup de 2-8x en CPUs multi-core.

**Implementación sugerida**:
```cpp
#pragma omp parallel for
for (size_t batch = 0; batch < num_batches; ++batch) {
    // Procesar batch independientemente
}
```

#### 3. Optimización de Memoria
- **Pool allocation**: Reducir fragmentación en creación frecuente de tensores
- **Memory mapping**: Para datasets grandes

#### 4. Soporte para GPU (CUDA)
- **Justificación**: Aceleración 10-1000x en entrenamiento de redes grandes
- **Implementación**: Kernels CUDA para operaciones matriciales

#### 5. Serialización de Modelos
- Guardar/cargar modelos entrenados
- Formato binario eficiente
- Compatibilidad entre versiones

---

## Trabajo en Equipo

| Tarea | Responsable | Rol |
|-------|-------------|-----|
| Investigación teórica | Marco Sabino Guardian | Estudio de redes neuronales |
| Diseño de la arquitectura | Marco Sabino Guardian | Diseño UML y patrones de diseño |
| Implementación del modelo | Marco Sabino Guardian | Código C++20 de la red neuronal |
| Pruebas y benchmarking | Marco Sabino Guardian | Generación de métricas y tests |
| Documentación y demo | Marco Sabino Guardian | Documentación completa y demostración |

---

## Conclusiones

### Logros Principales

- Implementación completa de red neuronal desde cero en C++20
- Álgebra genérica con tensores multidimensionales
- 5 funciones de activación diferentes
- 3 optimizadores (SGD, Adam, RMSprop)
- 3 aplicaciones prácticas funcionales
- Tests automatizados completos

### Aprendizajes

- Profundización en backpropagation y redes neuronales
- Dominio de C++20 (templates, genéricos, modern idioms)
- Patrones de diseño (Strategy, Factory, Template)

### Recomendaciones

- Paralelización con OpenMP para mejor rendimiento
- Soporte para GPU con CUDA
- Serialización de modelos entrenados

---

## Bibliografía

[1] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, May 2015. doi: 10.1038/nature14539.

[2] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, San Diego, CA, USA, 2015. arXiv: 1412.6980.

[3] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016. ISBN: 978-0262035613.

[4] R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training recurrent neural networks," in *Proc. 30th Int. Conf. Mach. Learn. (ICML)*, Atlanta, GA, USA, 2013, pp. 1310–1318.

[5] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proc. 13th Int. Conf. Artif. Intell. Stat. (AISTATS)*, Sardinia, Italy, 2010, pp. 249–256.

---


