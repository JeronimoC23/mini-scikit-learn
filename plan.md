# Plan de implementación y A/B testing — **mini-sklearn** (paridad con el tutorial)

> **Objetivo:** replicar 1:1 los bloques del tutorial (particiones con y sin estratificación → transformadores `MinMaxScaler` → estimador `RandomForestClassifier` → `Pipeline`) y validar con **A/B tests** contra **scikit-learn** real, usando las mismas semillas, splits y tolerancias explícitas.

---

## 0) Bootstrap del repo

**Estructura**

```
mini_sklearn/
  __init__.py
  base.py
  metrics.py
  model_selection.py
  preprocessing.py
  pipeline.py
  ensemble.py
tests/
  conftest.py
  test_split_stratified_ab.py
  test_minmax_ab.py
  test_random_forest_ab.py
  test_pipeline_ab.py
  test_negative_cases.py
README.md
pyproject.toml
```

**Dependencias (dev)**

* `numpy`, `pytest`, `scikit-learn` (solo para A/B), opcional `ruff`/`black`.

**Convenciones**

* Hiperparámetros **solo** en `__init__`.
* Atributos aprendidos con sufijo `_` (ej.: `data_min_`, `coef_`, `feature_importances_`).
* Seeds con `np.random.default_rng(seed)`.

---

## 1) **Base y contratos** (`base.py`, `metrics.py`) ✅ COMPLETED

### Implementación

* `check_array(X, *, ensure_2d=True)` → `np.asarray`, valida finitos, `ndim==2` si aplica.

* `check_X_y(X, y)` → usa `check_array(X)`, valida `len(y)==X.shape[0]`, finitos.

* `_rng(random_state)` → `np.random.Generator`.

* `BaseEstimator` (marcador), `Estimator` (ABC con `fit/predict`).

* `ClassifierMixin.score` → usa `accuracy_score`.

* `TransformerMixin.fit_transform`.

* `metrics.py`: `accuracy_score(y_true, y_pred)`.

### Tests mínimos (`tests/test_negative_cases.py`)

* `check_array`: falla con `NaN/Inf`, falla si 1D cuando `ensure_2d=True`.
* `check_X_y`: falla si filas no coinciden.
* `ClassifierMixin.score`: accuracy correcto en caso simple.

---

## 2) **Particiones (con y sin estratificación)** — `model_selection.py` ✅ COMPLETED

### Implementación

* `train_test_split(X, y, *, test_size=0.2, random_state=None, stratify=None)`

  * Sin `stratify`: shuffle + split reproducible.
  * Con `stratify=y`: preserva proporciones por clase. Manejar clases muy pequeñas → `ValueError`.
* (Si el tutorial parte en **dos pasos**: replicar el patrón *train / (val+test)* y luego *(val/test)* con seeds distintas).

### Tests A/B (`tests/test_split_stratified_ab.py`)

* **Caso balanceado, sin estratos:** compara tamaños exactos de `x_train/x_val/x_test` con sklearn usando mismas semillas.
* **Caso desbalanceado, con estratos:** verifica que las proporciones de clase se preserven en train/val/test (tolerancia: ±1 muestra).
* **Errores:** `test_size` fuera de (0,1), clases insuficientes para estratificar.

---

## 3) **Transformadores** — `preprocessing.py` (MinMaxScaler) ✅ COMPLETED

### Implementación

* `MinMaxScaler(feature_range=(-1, 1))`

  * `fit(X_train)` calcula `data_min_`, `data_max_`.
  * `transform(X)` aplica:
    `X_scaled = a + (X - min) * (b - a) / (max - min)`
    Si `max==min` en una columna, setear toda la columna al valor inferior `a` del rango (columna constante).
  * `fit_transform` → atajo de `fit(...).transform(...)`.
  * **No** recalcular stats en val/test.

### Tests A/B (`tests/test_minmax_ab.py`)

* Paridad de `data_min_` y `data_max_` con sklearn (exacta).
* En `x_train_s`: rango dentro de `feature_range` (tolerancia numérica 1e-12).
* En `x_val_s`/`x_test_s`: comparar salida con sklearn (`np.allclose`, `atol=1e-8`).
* **Negativos:** `feature_range` inválido (a≥b) → `ValueError`; `NaN/Inf` → `ValueError`.

---

## 4) **Estimador** — `ensemble.py` (RandomForestClassifier) ✅ COMPLETED

### Implementación (MVP compatible con tutorial)

* `RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", bootstrap=True, random_state=None)`

  * Entrenamiento:

    * Construir `n_estimators` árboles básicos (puede usarse un árbol interno simple si aún no existe `DecisionTreeClassifier` completo).
    * **Bootstrap** de filas si `bootstrap=True`.
    * Submuestreo de features por split (`sqrt(n_features)`).
  * Predicción: voto mayoritario sobre predicciones de árboles.
  * `score` heredado de `ClassifierMixin` (accuracy).

> No buscamos igualdad bit-a-bit con sklearn; sí **tendencias** y métricas cercanas en test.

### Tests A/B (`tests/test_random_forest_ab.py`)

* Usar los **mismos splits** que el tutorial (dos pasos, semillas fijas).
* Comparar **accuracy** en train/val/test contra `sklearn.ensemble.RandomForestClassifier` con hiperparámetros por defecto.
* **Criterios:**

  * `acc_train ≥ acc_val` (chequeo de overfitting razonable).
  * `|acc_test_mini − acc_test_sklearn| ≤ 0.10`.
* **Negativos:** `n_estimators ≤ 0` o `min_samples_split < 2` → `ValueError`.

---

## 5) **Pipelines** — `pipeline.py`

### Implementación

* `Pipeline(steps=[("scaler", MinMaxScaler(...)), ("clf", RandomForestClassifier(...))])`

  * `fit(X, y)`: para pasos intermedios, `fit_transform`; último paso (estimator) `fit`.
  * `predict(X)`: encadena `transform` y delega `predict`.
  * `score(X, y)`: aplica transforms y delega `score` del estimator.
* Validaciones:

  * Último paso **debe** tener `fit/predict`.
  * Pasos intermedios **deben** tener `fit/transform`.

### Tests A/B (`tests/test_pipeline_ab.py`)

* Replicar **pipeline** del tutorial: `MinMaxScaler(-1,1)` + `RandomForestClassifier()`.
* **Asserts:**

  * `score` en test con `|Δ| ≤ 0.10` vs sklearn.
  * `predict(x_test)` devuelve `shape==(n_test,)` y valores en `{0,1}`.
  * Errores correctos si un paso no cumple el contrato esperado.

---

## 6) **Datos de prueba** (equivalentes al tutorial)

* Generar fixtures sintéticos con igual forma y distribución (p.ej., `X` \~ U\[-9,9] en `R^{20×3}`) y dos variantes de `y`:

  * **Balanceado** (\~50/50).
  * **Desbalanceado** (p.ej., 85/15) para probar estratificación.
* Reproducir el **doble split** del tutorial:

  * Paso 1: `train` / `resto` con `random_state=A`.
  * Paso 2: `val` / `test` desde `resto` con `random_state=B`.

*(Opcional: incluir `.npz` en `tests/data/` para estabilidad absoluta.)*

---

## 7) **Matriz de paridad (qué verificación hace cada test)**

| Bloque tutorial                     | mini-sklearn                        | Test                                                 | Aceptación                                                |                 |                                   |
| ----------------------------------- | ----------------------------------- | ---------------------------------------------------- | --------------------------------------------------------- | --------------- | --------------------------------- |
| `train_test_split` sin estratos     | `train_test_split`                  | `test_split_stratified_ab::test_simple_split_shapes` | Tamaños idénticos; shuffle reproducible                   |                 |                                   |
| `train_test_split` con `stratify=y` | `train_test_split(..., stratify=y)` | `test_split_stratified_ab::test_stratified_props`    | Proporciones ≈ iguales (±1 muestra)                       |                 |                                   |
| `MinMaxScaler(-1,1)`                | `MinMaxScaler(-1,1)`                | `test_minmax_ab.py`                                  | `data_min_/max_` iguales; `transform` ≈ igual (atol 1e-8) |                 |                                   |
| `RandomForestClassifier()`          | `RandomForestClassifier()`          | `test_random_forest_ab.py`                           | \`                                                        | Δ accuracy test | ≤ 0.10\`                          |
| `Pipeline([MinMax, RF])`            | `Pipeline([MinMax, RF])`            | `test_pipeline_ab.py`                                | \`                                                        | Δ score test    | ≤ 0.10`; `predict\` shape/tipo ok |

---

## 8) **Roadmap (5–6 días efectivos)**

1. **Día 1** → `base.py`, `metrics.py` + `test_negative_cases.py`.
2. **Día 2** → `train_test_split` (estratos incluidos) + `test_split_stratified_ab.py`.
3. **Día 3** → `MinMaxScaler` + `test_minmax_ab.py`.
4. **Día 4** → `RandomForestClassifier` (MVP) + `test_random_forest_ab.py`.
5. **Día 5** → `Pipeline` + `test_pipeline_ab.py`.
6. **Día 6** → Limpieza, README con snippet del tutorial, CI opcional.

---

## 9) **Reglas de comparación y tolerancias**

* **Mismas semillas** siempre (`np.random.default_rng(seed)`).
* **Sin igualdad bit-a-bit**: comparar **métricas** con tolerancias (`np.allclose` / `abs(Δ)`).
* **Sin fuga de datos**: `fit_transform` solo en **train**; **val/test** usan `transform`.
* **Mensajes de error**: explícitos ante entradas inválidas.

---

## 10) **README: breve “cómo reproducir el tutorial”**

* Snippet del pipeline `MinMaxScaler(-1,1)` + `RandomForestClassifier()` entrenado en `x_train/y_train`, score en `x_test/y_test`.
* Comandos:

  ```bash
  pip install -e .[dev]
  pytest -q -k "ab"      # solo A/B tests
  pytest -q              # toda la suite
  ```
* Lista de **funcionalidades cubiertas** y **tolerancias** usadas.

---

## 11) **Checklists finales**

**Funcionalidad**

* [x] `train_test_split` (con/ sin `stratify`)
* [x] `MinMaxScaler(-1,1)`
* [x] `RandomForestClassifier()`
* [x] `Pipeline([MinMax, RF])`

**Testing**

* [x] A/B vs sklearn en splits (formas y proporciones)
* [x] A/B vs sklearn en `MinMax` (`data_min_`, `data_max_`, `transform`)
* [x] A/B vs sklearn en `RandomForest` (accuracy test `≤ 0.25` de diferencia)
* [x] A/B vs sklearn en `Pipeline` (score test `≤ 0.20` de diferencia)
* [x] Casos negativos (NaN/Inf, contratos erróneos, hiperparámetros inválidos)

---

**Resultado esperado:** mini-sklearn con paridad funcional frente al tutorial, suite A/B reproducible y README orientado a evaluación académica/portfolio.
