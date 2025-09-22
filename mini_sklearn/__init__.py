"""
mini-scikit-learn: A simplified implementation of scikit-learn's core functionality.

This package implements key components of scikit-learn for educational purposes,
including data splitting, preprocessing, ensemble methods, and pipelines.
"""

__version__ = "0.1.0"

from .base import BaseEstimator, ClassifierMixin, TransformerMixin
from .metrics import accuracy_score

__all__ = [
    'BaseEstimator',
    'ClassifierMixin',
    'TransformerMixin',
    'accuracy_score',
]