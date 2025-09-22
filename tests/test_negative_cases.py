"""
Tests for negative cases and edge conditions.

This module tests error handling and validation for various components.
"""

import pytest
import numpy as np
from mini_sklearn.base import check_array, check_X_y, ClassifierMixin
from mini_sklearn.metrics import accuracy_score


class DummyClassifier(ClassifierMixin):
    """Dummy classifier for testing ClassifierMixin."""

    def predict(self, X):
        # Always predict class 0
        return np.zeros(X.shape[0], dtype=int)


class TestCheckArray:
    """Test check_array function."""

    def test_check_array_with_nan(self):
        """Test that check_array raises ValueError for NaN values."""
        X_with_nan = np.array([[1, 2], [3, np.nan]])
        with pytest.raises(ValueError, match="Input contains NaN or infinite values"):
            check_array(X_with_nan)

    def test_check_array_with_inf(self):
        """Test that check_array raises ValueError for infinite values."""
        X_with_inf = np.array([[1, 2], [3, np.inf]])
        with pytest.raises(ValueError, match="Input contains NaN or infinite values"):
            check_array(X_with_inf)

    def test_check_array_1d_when_2d_required(self):
        """Test that check_array raises ValueError for 1D array when 2D required."""
        X_1d = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
            check_array(X_1d, ensure_2d=True)

    def test_check_array_1d_when_2d_not_required(self):
        """Test that check_array accepts 1D array when 2D not required."""
        X_1d = np.array([1, 2, 3])
        result = check_array(X_1d, ensure_2d=False)
        np.testing.assert_array_equal(result, X_1d)

    def test_check_array_valid_2d(self):
        """Test that check_array accepts valid 2D array."""
        X_2d = np.array([[1, 2], [3, 4]])
        result = check_array(X_2d)
        np.testing.assert_array_equal(result, X_2d)


class TestCheckXy:
    """Test check_X_y function."""

    def test_check_X_y_inconsistent_samples(self):
        """Test that check_X_y raises ValueError for inconsistent sample numbers."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # One extra sample
        with pytest.raises(ValueError, match="X and y have inconsistent numbers of samples"):
            check_X_y(X, y)

    def test_check_X_y_with_nan_in_y(self):
        """Test that check_X_y raises ValueError for NaN in target."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, np.nan])
        with pytest.raises(ValueError, match="Target array contains NaN or infinite values"):
            check_X_y(X, y)

    def test_check_X_y_valid(self):
        """Test that check_X_y accepts valid inputs."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_checked, y_checked = check_X_y(X, y)
        np.testing.assert_array_equal(X_checked, X)
        np.testing.assert_array_equal(y_checked, y)


class TestClassifierMixin:
    """Test ClassifierMixin functionality."""

    def test_classifier_mixin_score(self, simple_classification_data):
        """Test that ClassifierMixin.score returns correct accuracy."""
        X, y = simple_classification_data
        classifier = DummyClassifier()

        # Since DummyClassifier always predicts 0, accuracy should be
        # the fraction of true labels that are 0
        expected_accuracy = np.mean(y == 0)
        actual_accuracy = classifier.score(X, y)

        assert actual_accuracy == expected_accuracy


class TestAccuracyScore:
    """Test accuracy_score function."""

    def test_accuracy_score_different_lengths(self):
        """Test that accuracy_score raises ValueError for different length arrays."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError, match="y_true and y_pred have different lengths"):
            accuracy_score(y_true, y_pred)

    def test_accuracy_score_empty_arrays(self):
        """Test accuracy_score with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        score = accuracy_score(y_true, y_pred)
        assert score == 0.0

    def test_accuracy_score_perfect_prediction(self):
        """Test accuracy_score with perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        score = accuracy_score(y_true, y_pred)
        assert score == 1.0

    def test_accuracy_score_partial_correct(self):
        """Test accuracy_score with partially correct predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])  # 3 out of 4 correct
        score = accuracy_score(y_true, y_pred)
        assert score == 0.75