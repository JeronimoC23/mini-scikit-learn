"""
Base classes and utilities for mini-scikit-learn.

This module provides the foundational classes and validation functions
that other components inherit from and use.
"""

import numpy as np
from abc import ABC, abstractmethod


def check_array(X, *, ensure_2d=True):
    """
    Input validation on an array, list, sparse matrix or similar.

    Parameters
    ----------
    X : array-like
        Input data to validate
    ensure_2d : bool, default=True
        Whether to ensure the array is 2D

    Returns
    -------
    X_converted : ndarray
        The converted and validated array

    Raises
    ------
    ValueError
        If the array contains NaN or infinite values, or if ensure_2d=True
        and the array is not 2D
    """
    X = np.asarray(X)

    # Check for NaN and infinite values
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")

    # Check dimensionality
    if ensure_2d and X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

    return X


def check_X_y(X, y):
    """
    Input validation for X and y arrays.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector

    Returns
    -------
    X_converted : ndarray
        The converted and validated feature matrix
    y_converted : ndarray
        The converted and validated target vector

    Raises
    ------
    ValueError
        If X and y have inconsistent numbers of samples, or if they
        contain NaN or infinite values
    """
    X = check_array(X, ensure_2d=True)
    y = np.asarray(y)

    # Check for NaN and infinite values in y
    if not np.isfinite(y).all():
        raise ValueError("Target array contains NaN or infinite values")

    # Check consistent number of samples
    if len(y) != X.shape[0]:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: "
            f"{X.shape[0]} != {len(y)}"
        )

    return X, y


def _rng(random_state):
    """
    Create a numpy random number generator.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controls the randomness

    Returns
    -------
    rng : numpy.random.Generator
        Random number generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class BaseEstimator:
    """
    Base class for all estimators in mini-scikit-learn.

    This is a marker class that provides a common interface for all estimators.
    """
    pass


class Estimator(BaseEstimator, ABC):
    """
    Abstract base class for estimators.

    All estimators should implement fit and predict methods.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the estimator to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        pass


class ClassifierMixin:
    """
    Mixin class for classifiers.

    Provides a default implementation of the score method using accuracy.
    """

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X

        Returns
        -------
        score : float
            Mean accuracy
        """
        from .metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class TransformerMixin:
    """
    Mixin class for transformers.

    Provides a default implementation of fit_transform.
    """

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like, default=None
            Target values (ignored)

        Returns
        -------
        X_transformed : ndarray
            Transformed data
        """
        return self.fit(X, y).transform(X)