"""
Pipeline para encadenar transformadores y estimadores.

Este módulo proporciona una clase Pipeline que permite la aplicación secuencial
de transformadores seguidos por un estimador final, similar a sklearn.pipeline.Pipeline.
"""

import numpy as np
from .base import BaseEstimator, check_array


class Pipeline(BaseEstimator):
    """
    Pipeline de transformaciones con un estimador final.

    Aplicar secuencialmente una lista de transformaciones y un estimador final.
    Los pasos intermedios del pipeline deben ser 'transformadores', es decir, deben
    implementar métodos fit y transform.
    El estimador final solo necesita implementar fit y predict.

    Parameters
    ----------
    steps : list of tuple
        Lista de tuplas (name, transform) (implementando fit/transform) que se
        encadenan, en el orden en que se encadenan, con el último objeto
        siendo un estimador.

    Attributes
    ----------
    steps : list of tuple
        La lista de tuplas (name, transform)
    named_steps : dict
        Diccionario de pasos con nombres como claves

    Examples
    --------
    >>> from mini_sklearn.pipeline import Pipeline
    >>> from mini_sklearn.preprocessing import MinMaxScaler
    >>> from mini_sklearn.ensemble import RandomForestClassifier
    >>> pipe = Pipeline([
    ...     ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    ...     ('clf', RandomForestClassifier(n_estimators=50))
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    """

    def __init__(self, steps):
        """
        Inicializar el Pipeline.

        Parameters
        ----------
        steps : list of tuple
            Lista de tuplas (name, estimator)
        """
        if not isinstance(steps, list):
            raise TypeError("steps should be a list of (name, estimator) tuples")

        if len(steps) == 0:
            raise ValueError("Pipeline requires at least one step")

        # Validar formato de steps
        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError(
                    "Each step should be a tuple of (name, estimator), "
                    f"got {type(step)}"
                )
            name, estimator = step
            if not isinstance(name, str):
                raise TypeError(f"Step name should be a string, got {type(name)}")

        # Validar que todos los pasos intermedios tengan fit y transform
        for name, estimator in steps[:-1]:
            if not (hasattr(estimator, 'fit') and hasattr(estimator, 'transform')):
                raise TypeError(
                    f"All intermediate steps should have 'fit' and 'transform' methods. "
                    f"'{name}' (type {type(estimator).__name__}) doesn't."
                )

        # Validar que el estimador final tenga fit y predict
        final_name, final_estimator = steps[-1]
        if not (hasattr(final_estimator, 'fit') and hasattr(final_estimator, 'predict')):
            raise TypeError(
                f"Last step should have 'fit' and 'predict' methods. "
                f"'{final_name}' (type {type(final_estimator).__name__}) doesn't."
            )

        self.steps = steps
        self.named_steps = dict(steps)

    def fit(self, X, y):
        """
        Ajustar el pipeline.

        Ajustar todas las transformaciones una tras otra y transformar los
        datos, luego ajustar los datos transformados usando el estimador final.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like of shape (n_samples,)
            Valores objetivo

        Returns
        -------
        self : Pipeline
            Este estimador
        """
        X = check_array(X, ensure_2d=True)
        y = np.asarray(y)

        # Validar que X e y tengan el mismo número de muestras
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: "
                f"{X.shape[0]} != {len(y)}"
            )

        X_transformed = X

        # Ajustar y transformar a través de todos los pasos intermedios
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.fit_transform(X_transformed, y)

        # Ajustar el estimador final
        final_name, final_estimator = self.steps[-1]
        final_estimator.fit(X_transformed, y)

        return self

    def predict(self, X):
        """
        Aplicar transformaciones a los datos, y predecir con el estimador final.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos para predecir

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Valores predichos
        """
        X = check_array(X, ensure_2d=True)
        X_transformed = X

        # Transformar a través de todos los pasos intermedios (¡usar transform, no fit_transform!)
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)

        # Predecir con el estimador final
        final_name, final_estimator = self.steps[-1]
        return final_estimator.predict(X_transformed)

    def score(self, X, y):
        """
        Aplicar transformaciones, y puntuar con el estimador final.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Muestras de prueba
        y : array-like of shape (n_samples,)
            Etiquetas verdaderas para X

        Returns
        -------
        score : float
            Puntaje del estimador final
        """
        X = check_array(X, ensure_2d=True)
        y = np.asarray(y)

        X_transformed = X

        # Transformar a través de todos los pasos intermedios
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)

        # Puntuar con el estimador final
        final_name, final_estimator = self.steps[-1]

        # Verificar si el estimador final tiene un método score
        if not hasattr(final_estimator, 'score'):
            raise AttributeError(
                f"Final estimator '{final_name}' does not implement score method"
            )

        return final_estimator.score(X_transformed, y)

    def __getitem__(self, index):
        """
        Obtener un paso por índice o nombre.

        Parameters
        ----------
        index : int or str
            Índice o nombre del paso

        Returns
        -------
        estimator : object
            El estimador en el índice dado o con el nombre dado
        """
        if isinstance(index, str):
            return self.named_steps[index]
        return self.steps[index][1]

    def __len__(self):
        """Retornar el número de pasos en el pipeline."""
        return len(self.steps)

    def __repr__(self):
        """Representación en cadena del Pipeline."""
        steps_str = ",\n    ".join(
            f"('{name}', {type(estimator).__name__}(...))"
            for name, estimator in self.steps
        )
        return f"Pipeline([\n    {steps_str}\n])"
