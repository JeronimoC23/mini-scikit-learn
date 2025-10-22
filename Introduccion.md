# Introducción

Basándome en el tutorial de *scikit-learn* [link al Colab](https://colab.research.google.com/drive/1e369NLzd2YlEG4c_4fyCNqnebIG-rGni#scrollTo=fhYpws-kWuj4), mi objetivo es **replicar algunas de sus funcionalidades principales** para entender en profundidad cómo funciona la librería y, al mismo tiempo, ejercitar conceptos clave de **Python** y **Programación Orientada a Objetos (POO)**.

La idea no es solo usar *scikit-learn*, sino **construir desde cero** versiones simplificadas de sus componentes principales, aplicando buenas prácticas de diseño, principios de POO y testeo sistemático mediante comparación **A/B** contra *scikit-learn* real.

---

## Funcionalidades a replicar

1. **Particiones de datos**
   - Implementar `train_test_split`, incluyendo opción de estratificación (`stratify`) para datasets desbalanceados.
   - Reproducibilidad con semillas fijas (`random_state`) para obtener siempre los mismos splits.

2. **Transformadores de datos**
   - Crear un `MinMaxScaler` propio que replique el comportamiento de *scikit-learn* (`feature_range=(-1,1)`), con métodos `fit`, `transform` y `fit_transform`.
   - Asegurar un flujo correcto: *fit en train* y *transform en val/test* para evitar fuga de datos (*data leakage*).

3. **Modelos de aprendizaje**
   - Implementar un `RandomForestClassifier` básico con **bagging**, **bootstrap** y voto mayoritario, compatible con nuestra API.

4. **Pipelines**
   - Encadenar transformadores y modelos en un flujo reproducible con `Pipeline`, respetando la misma interfaz que *scikit-learn*.

5. **Métricas y evaluación**
   - Incluir métricas como `accuracy_score` para evaluar predicciones en train/val/test.

---

## Conceptos de Python y POO a trabajar

- **POO en Python**
  - Clases, herencia, mixins (`ClassifierMixin`, `TransformerMixin`), y principios como *Single Responsibility* y *Open/Closed*.
  - Clases abstractas (`ABC`) para definir contratos (`fit`, `predict`, `score`).

- **Diseño orientado a interfaces**
  - Separar transformadores, estimadores y pipelines con métodos claros y estados (`fit` guarda parámetros aprendidos en atributos con sufijo `_`).

- **Programación funcional y reproducibilidad**
  - Uso de `numpy` para cálculos vectorizados.
  - Control de aleatoriedad con `np.random.default_rng`.

- **Pruebas y validación**
  - Comparación **A/B** contra *scikit-learn* real con tolerancias definidas.
  - Tests para entradas inválidas (NaN, shapes incorrectos, hiperparámetros inválidos).

---

Este proyecto no solo busca entender cómo funcionan las herramientas más usadas en ciencia de datos, sino también **profundizar en patrones de diseño, buenas prácticas de programación y reproducibilidad**, habilidades esenciales para desarrollar librerías y herramientas profesionales.
