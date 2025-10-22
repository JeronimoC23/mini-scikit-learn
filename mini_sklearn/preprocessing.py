"""
Utilidades de preprocesamiento de datos.

Este módulo proporciona transformadores para escalar y normalizar datos.
"""

import numpy as np
from .base import BaseEstimator, TransformerMixin, check_array


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Transformar características escalando cada característica a un rango dado.

    Este estimador escala y traslada cada característica individualmente de tal manera
    que esté en el rango dado en el conjunto de entrenamiento, por ejemplo, entre
    cero y uno.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, 1)
        Rango deseado de los datos transformados

    Attributes
    ----------
    data_min_ : ndarray of shape (n_features,)
        Mínimo por característica visto en los datos
    data_max_ : ndarray of shape (n_features,)
        Máximo por característica visto en los datos
    scale_ : ndarray of shape (n_features,)
        Escalado relativo por característica de los datos
    """

    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """
        Calcular el mínimo y máximo a ser usado para el escalado posterior.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Los datos usados para calcular el mínimo y máximo por característica
            usado para el escalado posterior a lo largo del eje de características
        y : array-like, default=None
            Ignorado. Este parámetro existe solo por compatibilidad

        Returns
        -------
        self : object
            Escalador ajustado
        """
        # Validar feature_range
        if len(self.feature_range) != 2:
            raise ValueError("feature_range should be a tuple of 2 values")

        feature_min, feature_max = self.feature_range
        if feature_min >= feature_max:
            raise ValueError(
                f"Minimum of feature_range should be smaller than maximum. "
                f"Got {self.feature_range}"
            )

        # Validar y convertir entrada
        X = check_array(X, ensure_2d=True)

        # Calcular mínimo y máximo para cada característica
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        # Calcular escala (manejar características constantes)
        data_range = self.data_max_ - self.data_min_
        feature_range_diff = feature_max - feature_min

        # Para características constantes (data_range == 0), la escala será 0
        # Esto se manejará en el método transform
        self.scale_ = np.where(
            data_range != 0,
            feature_range_diff / data_range,
            0.0
        )

        return self

    def transform(self, X):
        """
        Escalar características de X según feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada que serán transformados

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Datos transformados
        """
        # Verificar si está ajustado
        if not hasattr(self, 'data_min_'):
            raise ValueError("This MinMaxScaler instance is not fitted yet.")

        # Validar y convertir entrada
        X = check_array(X, ensure_2d=True)

        if X.shape[1] != len(self.data_min_):
            raise ValueError(
                f"X has {X.shape[1]} features, but MinMaxScaler is expecting "
                f"{len(self.data_min_)} features"
            )

        feature_min, feature_max = self.feature_range

        # Aplicar transformación: X_scaled = feature_min + (X - data_min) * scale
        X_transformed = feature_min + (X - self.data_min_) * self.scale_

        # Manejar características constantes: establecerlas a feature_min
        constant_features = self.scale_ == 0
        if np.any(constant_features):
            X_transformed[:, constant_features] = feature_min

        return X_transformed

    def inverse_transform(self, X):
        """
        Deshacer el escalado de X según feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrada que serán transformados de vuelta

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Datos transformados en escala original
        """
        # Verificar si está ajustado
        if not hasattr(self, 'data_min_'):
            raise ValueError("This MinMaxScaler instance is not fitted yet.")

        # Validar y convertir entrada
        X = check_array(X, ensure_2d=True)

        if X.shape[1] != len(self.data_min_):
            raise ValueError(
                f"X has {X.shape[1]} features, but MinMaxScaler is expecting "
                f"{len(self.data_min_)} features"
            )

        feature_min, feature_max = self.feature_range

        # Transformación inversa: X_original = data_min + (X - feature_min) / scale
        X_original = np.zeros_like(X)

        # Manejar características no constantes
        non_constant = self.scale_ != 0
        if np.any(non_constant):
            X_original[:, non_constant] = (
                self.data_min_[non_constant] +
                (X[:, non_constant] - feature_min) / self.scale_[non_constant]
            )

        # Manejar características constantes: establecerlas a su valor constante original
        constant_features = self.scale_ == 0
        if np.any(constant_features):
            X_original[:, constant_features] = self.data_min_[constant_features]

        return X_original