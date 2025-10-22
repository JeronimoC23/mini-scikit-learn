"""
A/B tests for MinMaxScaler functionality against scikit-learn.

This module compares our implementation of MinMaxScaler with
scikit-learn's implementation to ensure functional parity.
"""

import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from mini_sklearn.preprocessing import MinMaxScaler


class TestMinMaxScalerAB:
    """A/B tests comparing mini_sklearn and sklearn MinMaxScaler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Create data with different scales and ranges
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [0.5, 1.5, 2.5]
        ])
        return X

    @pytest.fixture
    def data_with_constants(self):
        """Create data with constant features."""
        X = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 5.0, 9.0],
            [4.0, 5.0, 12.0],
        ])
        return X

    def test_default_feature_range_exact_match(self, sample_data):
        """Test that data_min_ and data_max_ match sklearn exactly."""
        X = sample_data

        # Our implementation
        scaler_mini = MinMaxScaler(feature_range=(-1, 1))
        scaler_mini.fit(X)

        # sklearn implementation
        scaler_sk = SklearnMinMaxScaler(feature_range=(-1, 1))
        scaler_sk.fit(X)

        # data_min_ and data_max_ should be exactly equal
        np.testing.assert_array_equal(scaler_mini.data_min_, scaler_sk.data_min_)
        np.testing.assert_array_equal(scaler_mini.data_max_, scaler_sk.data_max_)

    def test_transform_train_data_range(self, sample_data):
        """Test that transformed training data is within feature_range."""
        X = sample_data
        feature_range = (-1, 1)

        scaler = MinMaxScaler(feature_range=feature_range)
        X_transformed = scaler.fit_transform(X)

        # Check that all values are within the specified range
        assert np.all(X_transformed >= feature_range[0])
        assert np.all(X_transformed <= feature_range[1])

        # Check that min and max are achieved (for non-constant features)
        for feature_idx in range(X.shape[1]):
            feature_col = X_transformed[:, feature_idx]
            # Min and max should be very close to feature_range bounds
            assert abs(np.min(feature_col) - feature_range[0]) < 1e-12
            assert abs(np.max(feature_col) - feature_range[1]) < 1e-12

    def test_transform_vs_sklearn_close_match(self, sample_data):
        """Test that transform output is very close to sklearn."""
        X = sample_data

        # Our implementation
        scaler_mini = MinMaxScaler(feature_range=(-1, 1))
        X_mini = scaler_mini.fit_transform(X)

        # sklearn implementation
        scaler_sk = SklearnMinMaxScaler(feature_range=(-1, 1))
        X_sk = scaler_sk.fit_transform(X)

        # Should be very close (allowing for numerical precision differences)
        np.testing.assert_allclose(X_mini, X_sk, atol=1e-8)

    def test_separate_validation_data(self, sample_data):
        """Test transform on separate validation data matches sklearn."""
        X_train = sample_data

        # Create validation data with different range
        np.random.seed(123)
        X_val = np.random.uniform(-2, 15, size=(3, 3))

        # Our implementation
        scaler_mini = MinMaxScaler(feature_range=(-1, 1))
        scaler_mini.fit(X_train)
        X_val_mini = scaler_mini.transform(X_val)

        # sklearn implementation
        scaler_sk = SklearnMinMaxScaler(feature_range=(-1, 1))
        scaler_sk.fit(X_train)
        X_val_sk = scaler_sk.transform(X_val)

        # Should be very close
        np.testing.assert_allclose(X_val_mini, X_val_sk, atol=1e-8)

    def test_constant_features_handling(self, data_with_constants):
        """Test that constant features are handled correctly."""
        X = data_with_constants
        feature_range = (-1, 1)

        # Our implementation
        scaler_mini = MinMaxScaler(feature_range=feature_range)
        X_mini = scaler_mini.fit_transform(X)

        # sklearn implementation
        scaler_sk = SklearnMinMaxScaler(feature_range=feature_range)
        X_sk = scaler_sk.fit_transform(X)

        # Should match sklearn behavior
        np.testing.assert_allclose(X_mini, X_sk, atol=1e-8)

        # Constant feature (column 1) should be set to feature_range[0]
        assert np.all(X_mini[:, 1] == feature_range[0])

    def test_different_feature_ranges(self, sample_data):
        """Test various feature ranges match sklearn."""
        X = sample_data

        for feature_range in [(0, 1), (-2, 2), (10, 20), (-5, 10)]:
            # Our implementation
            scaler_mini = MinMaxScaler(feature_range=feature_range)
            X_mini = scaler_mini.fit_transform(X)

            # sklearn implementation
            scaler_sk = SklearnMinMaxScaler(feature_range=feature_range)
            X_sk = scaler_sk.fit_transform(X)

            # Should be very close
            np.testing.assert_allclose(X_mini, X_sk, atol=1e-8)

            # Check range bounds are respected
            assert np.all(X_mini >= feature_range[0])
            assert np.all(X_mini <= feature_range[1])

    def test_inverse_transform(self, sample_data):
        """Test that inverse_transform recovers original data."""
        X = sample_data

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_transformed = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_transformed)

        # Should recover original data (within numerical precision)
        np.testing.assert_allclose(X, X_recovered, atol=1e-10)

    def test_inverse_transform_vs_sklearn(self, sample_data):
        """Test that inverse_transform matches sklearn."""
        X = sample_data

        # Our implementation
        scaler_mini = MinMaxScaler(feature_range=(-1, 1))
        X_transformed_mini = scaler_mini.fit_transform(X)
        X_recovered_mini = scaler_mini.inverse_transform(X_transformed_mini)

        # sklearn implementation
        scaler_sk = SklearnMinMaxScaler(feature_range=(-1, 1))
        X_transformed_sk = scaler_sk.fit_transform(X)
        X_recovered_sk = scaler_sk.inverse_transform(X_transformed_sk)

        # Should be very close
        np.testing.assert_allclose(X_recovered_mini, X_recovered_sk, atol=1e-8)

    def test_error_cases(self, sample_data):
        """Test error handling."""
        X = sample_data

        # Invalid feature_range
        with pytest.raises(ValueError, match="Minimum of feature_range should be smaller"):
            scaler = MinMaxScaler(feature_range=(1, 1))
            scaler.fit(X)

        with pytest.raises(ValueError, match="Minimum of feature_range should be smaller"):
            scaler = MinMaxScaler(feature_range=(2, 1))
            scaler.fit(X)

        # Transform before fit
        scaler = MinMaxScaler()
        with pytest.raises(ValueError, match="This MinMaxScaler instance is not fitted"):
            scaler.transform(X)

        # Wrong number of features
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_wrong = np.array([[1, 2]])  # Wrong number of features
        with pytest.raises(ValueError, match="X has .* features, but MinMaxScaler is expecting"):
            scaler.transform(X_wrong)

    def test_fit_transform_equivalence(self, sample_data):
        """Test that fit_transform gives same result as fit().transform()."""
        X = sample_data

        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        X_transformed1 = scaler1.fit_transform(X)

        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        scaler2.fit(X)
        X_transformed2 = scaler2.transform(X)

        np.testing.assert_array_equal(X_transformed1, X_transformed2)

    def test_reproducibility(self, sample_data):
        """Test that repeated fits produce identical results."""
        X = sample_data

        # First fit
        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        X_transformed1 = scaler1.fit_transform(X)

        # Second fit
        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        X_transformed2 = scaler2.fit_transform(X)

        # Should be identical
        np.testing.assert_array_equal(X_transformed1, X_transformed2)
        np.testing.assert_array_equal(scaler1.data_min_, scaler2.data_min_)
        np.testing.assert_array_equal(scaler1.data_max_, scaler2.data_max_)