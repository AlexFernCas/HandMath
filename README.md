# 🧮 Clasificador de Operaciones Matemáticas con TensorFlow

Este proyecto implementa un modelo de red neuronal capaz de clasificar y realizar operaciones matemáticas simples en sus respectivas categorías (suma, resta, división, multiplicación) identificando la posición de la mano e interpretándola como un número.

Incluye todo el flujo: desde la preparación del dataset, la normalización de datos, entrenamiento del modelo y evaluación de resultados.

---

# 🎥 Demo

### Paso 1. Creación de datos de entrenamiento:

Ejecución del script capture_data.py.

Flujo:

* Fase 1: Posicionar la mano mostrando un número.

* Fase 2: Pulsar la tecla que corresponde al número para generar la etiqueta.

* Fase 3: Repetir fase 1 y 2 tantas veces como se desee.

* Fase 4: Pulsar tecla "q" para finalizar ejecución y generar archivo de datos.

![Creación de datos](./media/capture_data.gif)

### Paso 2. Normalización de datos

Ejecución de normalize.py.

Flujo:

* Fase 1: Convertir puntos a matriz (21 puntos x 3 coordenadas)

* Fase 2: Normalizar esos puntos. 

* Fase 3: Convertir la matriz a vector.


### Paso 3. Entrenamiento red neuronal:

Ejecución del script neuronal_network.py. 

![Entrenamiento modelo](./media/neuronal_network.gif)


### Paso 4. Demostración en tiempo real: 

Ejecución script hand_calc.py.

Flujo:

* Fase 1: muestra primer número con la mano → ENTR (confirmar).

* Fase 2: seleccionar operación con la mano (1:+ 2:- 3:/ 4:*) → ENTR.

* Fase 3: mostrar segundo número → ENTR.

* Fase 4: resultado en pantalla. ENTR para reiniciar.

![Demostración modelo](./media/hand_calc.gif)

---

# 📌 Características principales

Entrenamiento con dataset propio generado programáticamente.

Normalización de datos para un mejor rendimiento del modelo.

Codificación one-hot de etiquetas para clasificación multiclase.

Implementado con TensorFlow y Keras.

Código modular y fácil de ampliar con más operaciones o complejidad.

---

# 📁 Estructura del proyecto

La carpeta principal incluye todo lo necesario para reproducir y entrenar el modelo:

src/capture_data.py → Generación de datos. Mostrar un número con la mano y pulsar su tecla correspondiente para generar la etiqueta.

src/normalize.py → Normalización y preparación de datos de entrada.

src/neuronal_network.py → Entrenamiento de red neuronal.

src/hand_calc.py → Detección en tiempo real de posición de la mano y operación matemática.

utils/ → Funciones auxiliares para carga de datos y procesamiento.

artifacts/scaler.save → Objeto guardado del escalador utilizado para normalización (para inferencia futura).

requirements.txt → Lista de dependencias necesarias.

---

# 🛠️ Instalación

### Crear entorno virtual con Anaconda (opcional pero recomendado)

conda create -n calc_nn python=3.10

conda activate calc_nn

### Instalar dependencias

pip install -r requirements.txt

---

# 🚀 Entrenamiento y Resultados

### El modelo se entrenó con:

Capa oculta: 64 neuronas (ReLU)

Capa de salida: Softmax para clasificación

Épocas: 50

Optimizer: Adam

Loss: Categorical Crossentropy

---

# 📊 Precisión final: 

96,92% de precisión en el conjunto de prueba.

El modelo muestra un alto rendimiento en la clasificación de operaciones matemáticas simples.

<img src="./media/results_test.png" controls width="600">

---

# 📌 Notas
El modelo se entrenó con capa oculta (ReLU) de 128 neuronas y 100 épocas, pero se detectó que se podía conseguir el mismo rendimiento con un modelo más eficiente como el que se especifica en el apartado Entrenamiento y pruebas.

El modelo y el escalador se guardan y se cargan para inferencia en tiempo real.

El proyecto no incluye los modelos, escaladores y datos generados.

---

# 📜 Licencia
Todos los derechos reservados.