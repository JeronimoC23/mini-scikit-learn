"""
Métricas para evaluar el rendimiento del modelo.

Este módulo proporciona funciones para calcular varias métricas de rendimiento
para tareas de clasificación y regresión.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Puntaje de precisión de clasificación.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Etiquetas de verdad (correctas)
    y_pred : array-like of shape (n_samples,)
        Etiquetas predichas

    Returns
    -------
    score : float
        Puntaje de precisión (fracción de muestras clasificadas correctamente)

    Raises
    ------
    ValueError
        Si y_true e y_pred tienen longitudes diferentes
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    if len(y_true) == 0:
        return 0.0

    return np.mean(y_true == y_pred)