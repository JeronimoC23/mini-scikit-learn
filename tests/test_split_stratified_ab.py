"""
A/B tests for train_test_split functionality against scikit-learn.

This module compares our implementation of train_test_split with
scikit-learn's implementation to ensure functional parity.
"""

import pytest
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from mini_sklearn.model_selection import train_test_split


class TestTrainTestSplitAB:
    """A/B tests comparing mini_sklearn and sklearn train_test_split."""

    @pytest.fixture
    def balanced_data(self):
        """Create balanced classification dataset."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        # Create balanced binary classification
        y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        return X, y

    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced classification dataset."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        # Create imbalanced binary classification (85/15 split)
        y = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        return X, y

    def test_simple_split_shapes(self, balanced_data):
        """Test that simple splits produce identical shapes with sklearn."""
        X, y = balanced_data
        random_state = 42
        test_size = 0.2

        # Our implementation
        X_train_mini, X_test_mini, y_train_mini, y_test_mini = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # sklearn implementation
        X_train_sk, X_test_sk, y_train_sk, y_test_sk = sklearn_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Check shapes are identical
        assert X_train_mini.shape == X_train_sk.shape
        assert X_test_mini.shape == X_test_sk.shape
        assert y_train_mini.shape == y_train_sk.shape
        assert y_test_mini.shape == y_test_sk.shape

        # Check total samples are preserved
        assert len(X_train_mini) + len(X_test_mini) == len(X)
        assert len(y_train_mini) + len(y_test_mini) == len(y)

    def test_simple_split_reproducibility(self, balanced_data):
        """Test that splits are reproducible with same random_state."""
        X, y = balanced_data
        random_state = 42
        test_size = 0.2

        # First split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split with same random_state
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_stratified_split_proportions(self, imbalanced_data):
        """Test that stratified splits preserve class proportions."""
        X, y = imbalanced_data
        random_state = 42
        test_size = 0.2

        # Get original class proportions
        unique_classes, class_counts = np.unique(y, return_counts=True)
        original_proportions = class_counts / len(y)

        # Our stratified split
        X_train_mini, X_test_mini, y_train_mini, y_test_mini = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # sklearn stratified split
        X_train_sk, X_test_sk, y_train_sk, y_test_sk = sklearn_train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Check shapes are identical
        assert X_train_mini.shape == X_train_sk.shape
        assert X_test_mini.shape == X_test_sk.shape

        # Check class proportions in our implementation
        for class_label in unique_classes:
            # Train proportions
            train_prop_mini = np.mean(y_train_mini == class_label)
            train_prop_orig = np.mean(y == class_label)

            # Test proportions
            test_prop_mini = np.mean(y_test_mini == class_label)
            test_prop_orig = np.mean(y == class_label)

            # Proportions should be close to original (within ±0.05 tolerance)
            assert abs(train_prop_mini - train_prop_orig) <= 0.05
            assert abs(test_prop_mini - test_prop_orig) <= 0.05

    def test_stratified_vs_sklearn_similar_proportions(self, imbalanced_data):
        """Test that our stratified split produces similar proportions to sklearn."""
        X, y = imbalanced_data
        random_state = 42
        test_size = 0.2

        # Our stratified split
        _, _, y_train_mini, y_test_mini = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # sklearn stratified split
        _, _, y_train_sk, y_test_sk = sklearn_train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        unique_classes = np.unique(y)

        # Compare proportions between our implementation and sklearn
        for class_label in unique_classes:
            train_prop_mini = np.mean(y_train_mini == class_label)
            train_prop_sk = np.mean(y_train_sk == class_label)

            test_prop_mini = np.mean(y_test_mini == class_label)
            test_prop_sk = np.mean(y_test_sk == class_label)

            # Proportions should be similar (within ±0.05 tolerance)
            assert abs(train_prop_mini - train_prop_sk) <= 0.05
            assert abs(test_prop_mini - test_prop_sk) <= 0.05

    def test_error_cases(self, balanced_data):
        """Test error handling matches expected behavior."""
        X, y = balanced_data

        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size should be between 0 and 1"):
            train_test_split(X, y, test_size=1.5)

        with pytest.raises(ValueError, match="test_size should be between 0 and 1"):
            train_test_split(X, y, test_size=0)

        # Test stratify with mismatched length
        wrong_stratify = np.array([0, 1])  # Wrong length
        with pytest.raises(ValueError, match="stratify has .* samples but X has"):
            train_test_split(X, y, stratify=wrong_stratify)

        # Test stratify with class having only 1 sample
        single_sample_stratify = np.concatenate([
            np.zeros(len(y) - 1),
            np.ones(1)  # Only 1 sample of class 1
        ])
        with pytest.raises(ValueError, match="least populated class.*has only 1 member"):
            train_test_split(X, y, stratify=single_sample_stratify)

    def test_different_test_sizes(self, balanced_data):
        """Test various test_size values."""
        X, y = balanced_data
        random_state = 42

        for test_size in [0.1, 0.2, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            expected_test_size = int(np.ceil(test_size * len(X)))
            expected_train_size = len(X) - expected_test_size

            assert len(X_test) == expected_test_size
            assert len(X_train) == expected_train_size
            assert len(y_test) == expected_test_size
            assert len(y_train) == expected_train_size