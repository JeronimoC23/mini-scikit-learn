"""
Métodos de ensamble para aprendizaje automático.

Este módulo implementa clasificadores de ensamble como Random Forest, que combinan
múltiples estimadores base para crear modelos predictivos más fuertes.

Resumen de Random Forest:
-------------------------
Random Forest es un método de ensamble que combina múltiples árboles de decisión usando:
1. Bootstrap Aggregating (Bagging): Cada árbol se entrena en una muestra bootstrap
2. Selección Aleatoria de Características: Cada división considera solo un subconjunto aleatorio de características
3. Votación por Mayoría: La predicción final es la moda de todas las predicciones de los árboles

Conceptos Clave Implementados:
1. **Muestreo Bootstrap**: Crear conjuntos de entrenamiento muestreando con reemplazo
2. **Submuestreo Aleatorio de Características**: Para cada división, solo considerar sqrt(n_features) características aleatorias
3. **Árboles de Decisión Simples**: Árboles básicos con divisiones aleatorias (no óptimas pero funcionales)
4. **Votación por Mayoría**: Agregar predicciones de todos los árboles
5. **Aleatoriedad Reproducible**: Usar el Generator de numpy para resultados consistentes

Esta implementación prioriza la claridad educativa sobre la optimización, enfocándose en
demostrar los conceptos centrales de Random Forest en lugar de lograr el máximo rendimiento.
"""

import numpy as np
from .base import BaseEstimator, ClassifierMixin, Estimator, check_X_y, check_array, _rng


class SimpleDecisionTree(Estimator):
    """
    Una implementación simple de árbol de decisión para usar dentro de Random Forest.

    Este es un árbol básico que hace divisiones aleatorias en lugar de óptimas.
    El propósito es proporcionar un árbol funcional para Random Forest sin
    implementar la complejidad completa de algoritmos óptimos de árboles de decisión.

    Algoritmo:
    1. En cada nodo, seleccionar aleatoriamente un subconjunto de características
    2. Para cada característica seleccionada, probar un umbral aleatorio
    3. Elegir la división que da la mejor reducción de impureza de Gini
    4. Construir recursivamente subárboles izquierdo y derecho
    5. Detenerse cuando se alcanza max_depth o el nodo es puro

    Parameters
    ----------
    max_depth : int or None, default=None
        Profundidad máxima del árbol
    min_samples_split : int, default=2
        Muestras mínimas requeridas para dividir un nodo
    max_features : int or None, default=None
        Número de características a considerar para cada división
    random_state : int or None, default=None
        Semilla aleatoria para reproducibilidad
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def _gini_impurity(self, y):
        """
        Calcular la impureza de Gini para un conjunto de etiquetas.

        La impureza de Gini mide qué tan "impuro" es un conjunto de etiquetas.
        Gini = 1 - sum(p_i^2) donde p_i es la proporción de la clase i

        Parameters
        ----------
        y : array-like
            Etiquetas

        Returns
        -------
        float
            Impureza de Gini (0 = puro, mayor = más mezclado)
        """
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def _best_split(self, X, y):
        """
        Encontrar la mejor división para el nodo actual.

        Esta es una versión simplificada que prueba umbrales aleatorios en lugar de
        encontrar la división óptima. Para cada característica, probamos algunos umbrales
        aleatorios y elegimos el que tiene la mejor mejora de Gini.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matriz de características
        y : array-like of shape (n_samples,)
            Etiquetas objetivo

        Returns
        -------
        dict or None
            Información de mejor división con claves: 'feature', 'threshold', 'impurity_reduction'
            Retorna None si no se encuentra una buena división
        """
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None

        # Determinar cuántas características considerar
        if self.max_features is None:
            max_features = n_features
        else:
            max_features = min(self.max_features, n_features)

        # Seleccionar aleatoriamente características a considerar
        rng = _rng(self.random_state)
        feature_indices = rng.choice(n_features, size=max_features, replace=False)

        best_split = None
        best_impurity_reduction = 0
        current_impurity = self._gini_impurity(y)

        # Probar cada característica seleccionada
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]

            # Saltar si todos los valores son iguales
            if np.all(feature_values == feature_values[0]):
                continue

            # Probar algunos umbrales aleatorios para esta característica
            min_val, max_val = feature_values.min(), feature_values.max()

            # Probar 5 umbrales aleatorios entre mín y máx
            for _ in range(5):
                threshold = rng.uniform(min_val, max_val)

                # Dividir datos basados en el umbral
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Saltar si la división crea particiones vacías
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Calcular Gini ponderado para esta división
                left_y, right_y = y[left_mask], y[right_mask]
                left_weight = len(left_y) / n_samples
                right_weight = len(right_y) / n_samples

                weighted_gini = (left_weight * self._gini_impurity(left_y) +
                               right_weight * self._gini_impurity(right_y))

                impurity_reduction = current_impurity - weighted_gini

                # Actualizar mejor división si esta es mejor
                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'impurity_reduction': impurity_reduction
                    }

        return best_split

    def _build_tree(self, X, y, depth=0):
        """
        Construir recursivamente el árbol de decisión.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matriz de características
        y : array-like of shape (n_samples,)
            Etiquetas objetivo
        depth : int, default=0
            Profundidad actual en el árbol

        Returns
        -------
        dict
            Nodo del árbol con estructura:
            - 'is_leaf': bool indicando si este es un nodo hoja
            - 'prediction': clase más común (para nodos hoja)
            - 'feature', 'threshold': criterios de división (para nodos internos)
            - 'left', 'right': nodos hijos (para nodos internos)
        """
        # Verificar criterios de parada
        classes, counts = np.unique(y, return_counts=True)
        most_common_class = classes[np.argmax(counts)]

        # Detener si: nodo puro, max_depth alcanzado, o no hay suficientes muestras
        if (len(classes) == 1 or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(y) < self.min_samples_split):
            return {
                'is_leaf': True,
                'prediction': most_common_class
            }

        # Encontrar mejor división
        split = self._best_split(X, y)
        if split is None:
            return {
                'is_leaf': True,
                'prediction': most_common_class
            }

        # Crear división
        feature_idx = split['feature']
        threshold = split['threshold']
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Construir recursivamente nodos hijos
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'is_leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_node,
            'right': right_node
        }

    def fit(self, X, y):
        """
        Ajustar el árbol de decisión a los datos de entrenamiento.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like of shape (n_samples,)
            Etiquetas objetivo

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        """
        Predecir una sola muestra recorriendo el árbol.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Muestra única
        node : dict
            Nodo actual del árbol

        Returns
        -------
        int
            Clase predicha
        """
        if node['is_leaf']:
            return node['prediction']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        """
        Hacer predicciones sobre datos nuevos.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Etiquetas predichas
        """
        X = check_array(X, ensure_2d=True)
        return np.array([self._predict_sample(x, self.tree_) for x in X])


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Una implementación de clasificador Random Forest.

    Random Forest construye múltiples árboles de decisión y combina sus predicciones
    mediante votación por mayoría. Cada árbol se entrena en una muestra bootstrap de los
    datos, y cada división considera solo un subconjunto aleatorio de características.

    Pasos del Algoritmo:
    1. **Muestreo Bootstrap**: Para cada árbol, crear un conjunto de entrenamiento muestreando
       con reemplazo de los datos originales (mismo tamaño que el original)
    2. **Submuestreo de Características**: En cada división en cada árbol, seleccionar aleatoriamente
       sqrt(n_features) características a considerar
    3. **Entrenamiento de Árboles**: Entrenar cada árbol en su muestra bootstrap usando solo
       las características seleccionadas en cada división
    4. **Predicción**: Para datos nuevos, obtener predicciones de todos los árboles y retornar
       la predicción más común (voto por mayoría)

    Beneficios de Random Forest:
    - **Reduce el Sobreajuste**: Los árboles individuales pueden sobreajustarse, pero el promedio
      reduce este efecto
    - **Maneja Valores Faltantes**: Puede trabajar con datos incompletos
    - **Importancia de Características**: Puede estimar qué características son más útiles
    - **Robusto**: Menos sensible a valores atípicos que árboles individuales

    Parameters
    ----------
    n_estimators : int, default=100
        Número de árboles en el bosque
    max_depth : int or None, default=None
        Profundidad máxima de cada árbol
    min_samples_split : int, default=2
        Muestras mínimas requeridas para dividir un nodo
    max_features : str or int or None, default="sqrt"
        Número de características a considerar para cada división:
        - "sqrt": sqrt(n_features)
        - int: número exacto
        - None: todas las características
    bootstrap : bool, default=True
        Si usar muestreo bootstrap
    random_state : int or None, default=None
        Semilla aleatoria para reproducibilidad

    Attributes
    ----------
    estimators_ : list of SimpleDecisionTree
        La colección de árboles de decisión ajustados
    n_classes_ : int
        Número de clases
    classes_ : array-like of shape (n_classes,)
        Etiquetas de clase
    feature_importances_ : array-like of shape (n_features,)
        Puntuaciones de importancia de características (no implementado en esta versión básica)
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _validate_parameters(self):
        """Validar hiperparámetros."""
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators should be > 0, got {self.n_estimators}")
        if self.min_samples_split < 2:
            raise ValueError(f"min_samples_split should be >= 2, got {self.min_samples_split}")

    def _get_max_features(self, n_features):
        """
        Determinar el número de características a usar para cada división.

        Parameters
        ----------
        n_features : int
            Número total de características

        Returns
        -------
        int
            Número de características a usar
        """
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _bootstrap_sample(self, X, y, rng):
        """
        Crear una muestra bootstrap de los datos.

        El muestreo bootstrap involucra muestrear con reemplazo para crear
        un nuevo conjunto de datos del mismo tamaño que el original. Esto introduce
        aleatoriedad y ayuda a reducir el sobreajuste.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matriz de características
        y : array-like of shape (n_samples,)
            Etiquetas objetivo
        rng : numpy.random.Generator
            Generador de números aleatorios

        Returns
        -------
        X_bootstrap : array-like of shape (n_samples, n_features)
            Muestra bootstrap de características
        y_bootstrap : array-like of shape (n_samples,)
            Muestra bootstrap de etiquetas
        """
        n_samples = X.shape[0]

        if self.bootstrap:
            # Muestrear con reemplazo (bootstrap)
            indices = rng.choice(n_samples, size=n_samples, replace=True)
        else:
            # Usar todas las muestras (sin bootstrap)
            indices = np.arange(n_samples)
            rng.shuffle(indices)

        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Ajustar el Random Forest a los datos de entrenamiento.

        Este método:
        1. Valida entradas y parámetros
        2. Crea n_estimators árboles de decisión
        3. Para cada árbol:
           a. Crea una muestra bootstrap
           b. Ajusta el árbol a la muestra bootstrap
           c. Almacena el árbol ajustado

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like of shape (n_samples,)
            Etiquetas objetivo

        Returns
        -------
        self
        """
        # Validar entradas
        X, y = check_X_y(X, y)
        self._validate_parameters()

        # Almacenar clases e información básica
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape

        # Determinar max_features para árboles
        max_features_per_tree = self._get_max_features(n_features)

        # Inicializar generador de números aleatorios
        rng = _rng(self.random_state)

        # Entrenar cada árbol
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Crear muestra bootstrap
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y, rng)

            # Crear y entrenar árbol
            # Usar estado aleatorio diferente para cada árbol para asegurar diversidad
            tree_random_state = rng.integers(0, 2**31) if self.random_state is not None else None

            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features_per_tree,
                random_state=tree_random_state
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Hacer predicciones usando votación por mayoría de todos los árboles.

        Algoritmo:
        1. Obtener predicciones de cada árbol en el bosque
        2. Para cada muestra, contar votos para cada clase
        3. Retornar la clase con más votos

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Etiquetas predichas
        """
        # Validar entrada
        X = check_array(X, ensure_2d=True)

        if not hasattr(self, 'estimators_'):
            raise ValueError("This RandomForestClassifier instance is not fitted yet.")

        n_samples = X.shape[0]

        # Recolectar predicciones de todos los árboles
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=int)

        for i, tree in enumerate(self.estimators_):
            all_predictions[:, i] = tree.predict(X)

        # Votación por mayoría: para cada muestra, encontrar la predicción más común
        final_predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Contar votos para cada clase
            votes = np.bincount(all_predictions[i, :], minlength=self.n_classes_)
            # Elegir clase con más votos
            final_predictions[i] = self.classes_[np.argmax(votes)]

        return final_predictions

    def predict_proba(self, X):
        """
        Predecir probabilidades de clase como la fracción de árboles votando por cada clase.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada

        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Probabilidades de clase
        """
        X = check_array(X, ensure_2d=True)

        if not hasattr(self, 'estimators_'):
            raise ValueError("This RandomForestClassifier instance is not fitted yet.")

        n_samples = X.shape[0]

        # Recolectar predicciones de todos los árboles
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=int)

        for i, tree in enumerate(self.estimators_):
            all_predictions[:, i] = tree.predict(X)

        # Calcular probabilidades como fracción de votos
        probabilities = np.zeros((n_samples, self.n_classes_))

        for i in range(n_samples):
            # Contar votos para cada clase
            votes = np.bincount(all_predictions[i, :], minlength=self.n_classes_)
            # Convertir a probabilidades
            probabilities[i, :] = votes / self.n_estimators

        return probabilities