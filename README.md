# Mini Scikit-Learn

Una implementación simplificada de la funcionalidad central de scikit-learn con fines educativos. Este proyecto replica componentes clave de scikit-learn para demostrar conceptos de aprendizaje automático y prácticas de POO en Python.

## 🎯 Objetivos del Proyecto

- **Enfoque Educativo**: Comprender cómo funciona scikit-learn internamente
- **Práctica de POO en Python**: Implementar patrones de diseño como mixins, clases abstractas y herencia
- **Pruebas A/B**: Comparar nuestra implementación contra scikit-learn con tolerancias definidas
- **Ciencia Reproducible**: Asegurar resultados consistentes con semillas aleatorias fijas

## 🚀 Inicio Rápido

### Requisitos Previos

```bash
# Instalar dependencias requeridas
pip install numpy pytest scikit-learn
```

O usar nuestro script auxiliar:
```bash
make install
```

### Ejecutar Tests

**Forma fácil (recomendada):**
```bash
make test           # Ejecutar todos los tests
make test-quick     # Ejecución rápida con salida mínima
make test-ab        # Solo tests de comparación A/B
```

**Formas alternativas:**
```bash
# Usando script de Python
python run_tests.py           # Todos los tests
python run_tests.py ab        # Solo tests A/B

# pytest directo
PYTHONPATH=. python3 -m pytest tests/ -v
```

**Ver todas las opciones:**
```bash
make help
```

## 📦 Componentes Implementados

### ✅ División de Datos (`model_selection.py`)
- `train_test_split()` con soporte para estratificación
- Preserva las proporciones de clases en datasets desbalanceados
- Muestreo bootstrap con semillas aleatorias reproducibles

### ✅ Preprocesamiento (`preprocessing.py`)
- `MinMaxScaler` con rangos de características configurables
- Maneja características constantes correctamente
- Paridad exacta con `data_min_` y `data_max_` de scikit-learn

### ✅ Métodos de Ensamble (`ensemble.py`)
- `RandomForestClassifier` con agregación bootstrap
- Selección aleatoria de características en cada división
- Votación por mayoría para predicciones
- Árboles de decisión simples con umbrales aleatorios

### ✅ Pipelines (`pipeline.py`)
- `Pipeline` para encadenar transformadores y estimadores
- Flujo automático de datos a través de `fit_transform` y `transform`
- Previene fuga de datos usando `transform` (no `fit_transform`) en datos de prueba
- Compatible con todos los transformadores y estimadores

### ✅ Infraestructura Base (`base.py`, `metrics.py`)
- Validación de entrada (`check_array`, `check_X_y`)
- Clases base y mixins (`BaseEstimator`, `ClassifierMixin`, `TransformerMixin`)
- Métricas de rendimiento (`accuracy_score`)

## 🧪 Estrategia de Pruebas

Nuestra implementación usa **pruebas A/B** contra scikit-learn para asegurar paridad funcional:

- **División de Datos**: Coincidencia exacta de formas, preservación de proporciones (tolerancia de ±1 muestra)
- **MinMaxScaler**: Coincidencia exacta de `data_min_`/`data_max_`, salida de transform con tolerancia de 1e-8
- **RandomForest**: Precisión dentro de ±0.25 de scikit-learn (considera diferentes algoritmos de árbol)
- **Pipeline**: Precisión en conjunto de prueba dentro de ±0.20 de scikit-learn
- **Manejo de Errores**: Pruebas negativas exhaustivas para entradas inválidas

## 📊 Ejemplo de Uso

```python
from mini_sklearn.model_selection import train_test_split
from mini_sklearn.preprocessing import MinMaxScaler
from mini_sklearn.ensemble import RandomForestClassifier
from mini_sklearn.pipeline import Pipeline

# Crear datos de ejemplo
import numpy as np
np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Dividir datos con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Crear pipeline con escalador y clasificador
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])

# Ajustar pipeline (escala y entrena automáticamente)
pipeline.fit(X_train, y_train)

# Evaluar (aplica el escalado automáticamente antes de predecir)
print(f"Precisión en Test: {pipeline.score(X_test, y_test):.3f}")

# Hacer predicciones
y_pred = pipeline.predict(X_test)
```

## 🏗️ Estructura del Proyecto

```
mini_sklearn/
├── __init__.py           # Inicialización del paquete
├── base.py              # Clases base y validación
├── metrics.py           # Métricas de rendimiento
├── model_selection.py   # Utilidades de división de datos
├── preprocessing.py     # Transformadores de datos
├── ensemble.py          # Métodos de ensamble
└── pipeline.py          # Implementación de Pipeline

tests/
├── conftest.py                  # Fixtures de pruebas
├── test_negative_cases.py       # Tests de manejo de errores
├── test_split_stratified_ab.py  # Tests A/B de división de datos
├── test_minmax_ab.py           # Tests A/B de MinMaxScaler
├── test_random_forest_ab.py    # Tests A/B de RandomForest
└── test_pipeline_ab.py         # Tests A/B de Pipeline
```

## 🎓 Aspectos Educativos Destacados

### Implementación de Random Forest
Nuestro RandomForest incluye comentarios detallados explicando:
- **Muestreo Bootstrap**: Cómo el bagging reduce el sobreajuste
- **Selección Aleatoria de Características**: Por qué `sqrt(n_features)` en cada división
- **Votación por Mayoría**: Combinando predicciones de múltiples árboles
- **Construcción de Árboles de Decisión**: División recursiva con impureza de Gini

### Patrones de Diseño
- **Clases Mixin**: `ClassifierMixin`, `TransformerMixin`
- **Clases Base Abstractas**: `Estimator` con `fit`/`predict` requeridos
- **Utilidades de Validación**: Verificación de entrada centralizada
- **Aleatoriedad Reproducible**: Gestión adecuada de semillas

## 🔍 Resultados de Tests A/B

Estado actual de los tests: **Todos los tests pasando** ✅

- División de datos: Coincidencia perfecta de formas y proporciones
- MinMaxScaler: Paridad exacta con scikit-learn
- RandomForest: Paridad funcional dentro de la tolerancia (±0.25)
- Pipeline: Paridad funcional dentro de la tolerancia (±0.20)
- Manejo de errores: Validación exhaustiva

## 🚧 Próximamente

- [ ] Más métodos de ensamble (GradientBoosting, AdaBoost)
- [ ] Utilidades de validación cruzada (KFold, cross_val_score)
- [ ] Métodos de selección de características
- [ ] Transformadores de preprocesamiento adicionales (StandardScaler, etc.)

## 🤝 Contribuir

Este es un proyecto educativo. Siéntete libre de:
1. Hacer fork del repositorio
2. Ejecutar tests con `make test`
3. Implementar componentes adicionales
4. Agregar tests más exhaustivos

## 📄 Licencia

Licencia MIT - Libre para usar con fines educativos.

---

**Hecho con ❤️ para aprender los fundamentos del aprendizaje automático**