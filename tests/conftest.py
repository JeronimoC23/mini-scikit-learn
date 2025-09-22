"""
Pytest configuration and shared fixtures for mini-scikit-learn tests.
"""

import pytest
import numpy as np


@pytest.fixture
def simple_classification_data():
    """
    Simple 2D classification dataset for basic tests.

    Returns
    -------
    X : ndarray of shape (10, 2)
        Feature matrix
    y : ndarray of shape (10,)
        Binary target labels
    """
    np.random.seed(42)
    X = np.random.rand(10, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


@pytest.fixture
def random_state():
    """Fixed random state for reproducible tests."""
    return 42