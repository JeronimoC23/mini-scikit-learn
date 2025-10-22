"""
Utilidades de división de datos para splits de entrenamiento/validación/prueba.

Este módulo proporciona funciones para dividir conjuntos de datos preservando
opcionalmente las distribuciones de clases (estratificación).
"""

import numpy as np
from .base import check_X_y, _rng


def train_test_split(X, y, *, test_size=0.2, random_state=None, stratify=None):
    """
    Dividir arrays en subconjuntos aleatorios de entrenamiento y prueba.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Datos de entrenamiento
    y : array-like of shape (n_samples,)
        Valores objetivo
    test_size : float, default=0.2
        Proporción del conjunto de datos a incluir en el split de prueba
    random_state : int, RandomState instance or None, default=None
        Controla la mezcla aplicada a los datos antes de aplicar el split
    stratify : array-like or None, default=None
        Si no es None, los datos se dividen de manera estratificada, usando esto
        como las etiquetas de clase

    Returns
    -------
    X_train : ndarray
        Datos de entrenamiento
    X_test : ndarray
        Datos de prueba
    y_train : ndarray
        Objetivos de entrenamiento
    y_test : ndarray
        Objetivos de prueba

    Raises
    ------
    ValueError
        Si test_size no está entre 0 y 1, o si la estratificación falla
        debido a muestras insuficientes por clase
    """
    # Validar entradas
    X, y = check_X_y(X, y)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size should be between 0 and 1, got {test_size}")

    n_samples = X.shape[0]
    n_test = int(np.ceil(test_size * n_samples))
    n_train = n_samples - n_test

    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"With n_samples={n_samples} and test_size={test_size}, "
            f"the resulting train size={n_train} or test size={n_test} is 0"
        )

    rng = _rng(random_state)

    if stratify is None:
        # División aleatoria simple
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # División estratificada
        stratify = np.asarray(stratify)
        if len(stratify) != n_samples:
            raise ValueError(
                f"stratify has {len(stratify)} samples but X has {n_samples}"
            )

        # Obtener clases únicas y sus conteos
        classes, y_indices = np.unique(stratify, return_inverse=True)
        n_classes = len(classes)

        # Calcular muestras por clase para el conjunto de prueba
        class_counts = np.bincount(y_indices)
        test_counts = np.round(class_counts * test_size).astype(int)

        # Asegurar que tengamos al menos 1 muestra por clase en prueba si la clase existe
        # y al menos 1 muestra por clase en entrenamiento
        for i, (count, test_count) in enumerate(zip(class_counts, test_counts)):
            if count == 1 and test_size > 0:
                raise ValueError(
                    f"The least populated class in stratify has only 1 member, "
                    f"which is too few. The minimum number of members in any "
                    f"class cannot be less than 2."
                )
            if test_count == 0 and count > 0:
                test_counts[i] = 1
            if test_count == count and count > 1:
                test_counts[i] = count - 1

        # Verificar si todavía tenemos splits válidos
        if test_counts.sum() > n_test:
            # Reducir conteos de prueba proporcionalmente
            excess = test_counts.sum() - n_test
            for i in range(len(test_counts)):
                if test_counts[i] > 1 and excess > 0:
                    reduction = min(test_counts[i] - 1, excess)
                    test_counts[i] -= reduction
                    excess -= reduction

        # Recolectar índices para cada clase
        train_indices = []
        test_indices = []

        for class_idx, (class_label, test_count) in enumerate(zip(classes, test_counts)):
            # Obtener todos los índices para esta clase
            class_mask = y_indices == class_idx
            class_indices = np.where(class_mask)[0]

            # Mezclar índices para esta clase
            rng.shuffle(class_indices)

            # Dividir en entrenamiento/prueba
            test_indices.extend(class_indices[:test_count])
            train_indices.extend(class_indices[test_count:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Mezcla final para combinar clases
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )