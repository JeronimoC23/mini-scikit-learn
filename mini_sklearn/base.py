"""
Clases base y utilidades para mini-scikit-learn.

Este módulo proporciona las clases fundamentales y funciones de validación
que otros componentes heredan y utilizan.
"""

import numpy as np
from abc import ABC, abstractmethod


def check_array(X, *, ensure_2d=True):
    """
    Validación de entrada para un array, lista, matriz dispersa o similar.

    Parameters
    ----------
    X : array-like
        Datos de entrada a validar
    ensure_2d : bool, default=True
        Si se debe asegurar que el array sea 2D

    Returns
    -------
    X_converted : ndarray
        El array convertido y validado

    Raises
    ------
    ValueError
        Si el array contiene valores NaN o infinitos, o si ensure_2d=True
        y el array no es 2D
    """
    X = np.asarray(X)

    # Verificar valores NaN e infinitos
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")

    # Verificar dimensionalidad
    if ensure_2d and X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

    return X


def check_X_y(X, y):
    """
    Validación de entrada para arrays X e y.

    Parameters
    ----------
    X : array-like
        Matriz de características
    y : array-like
        Vector objetivo

    Returns
    -------
    X_converted : ndarray
        La matriz de características convertida y validada
    y_converted : ndarray
        El vector objetivo convertido y validado

    Raises
    ------
    ValueError
        Si X e y tienen números inconsistentes de muestras, o si
        contienen valores NaN o infinitos
    """
    X = check_array(X, ensure_2d=True)
    y = np.asarray(y)

    # Verificar valores NaN e infinitos en y
    if not np.isfinite(y).all():
        raise ValueError("Target array contains NaN or infinite values")

    # Verificar número consistente de muestras
    if len(y) != X.shape[0]:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: "
            f"{X.shape[0]} != {len(y)}"
        )

    return X, y


def _rng(random_state):
    """
    Crear un generador de números aleatorios de numpy.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controla la aleatoriedad

    Returns
    -------
    rng : numpy.random.Generator
        Generador de números aleatorios
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class BaseEstimator:
    """
    Clase base para todos los estimadores en mini-scikit-learn.

    Esta es una clase marcadora que proporciona una interfaz común para todos los estimadores.
    """
    pass


class Estimator(BaseEstimator, ABC):
    """
    Clase base abstracta para estimadores.

    Todos los estimadores deben implementar los métodos fit y predict.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Ajustar el estimador a los datos de entrenamiento.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like of shape (n_samples,)
            Valores objetivo

        Returns
        -------
        self : object
            Retorna la instancia misma
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Hacer predicciones sobre datos nuevos.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Valores predichos
        """
        pass


class ClassifierMixin:
    """
    Clase mixin para clasificadores.

    Proporciona una implementación por defecto del método score usando precisión.
    """

    def score(self, X, y):
        """
        Retornar la precisión promedio en los datos de prueba y etiquetas dados.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Muestras de prueba
        y : array-like of shape (n_samples,)
            Etiquetas verdaderas para X

        Returns
        -------
        score : float
            Precisión promedio
        """
        from .metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class TransformerMixin:
    """
    Clase mixin para transformadores.

    Proporciona una implementación por defecto de fit_transform.
    """

    def fit_transform(self, X, y=None):
        """
        Ajustar a los datos, luego transformarlos.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada
        y : array-like, default=None
            Valores objetivo (ignorados)

        Returns
        -------
        X_transformed : ndarray
            Datos transformados
        """
        return self.fit(X, y).transform(X)