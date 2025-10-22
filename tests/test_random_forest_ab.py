"""
A/B tests for RandomForestClassifier functionality against scikit-learn.

This module compares our implementation of RandomForestClassifier with
scikit-learn's implementation to ensure functional behavior, though not
necessarily identical predictions due to different tree-building algorithms.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.datasets import make_classification
from mini_sklearn.ensemble import RandomForestClassifier


class TestRandomForestClassifierAB:
    """A/B tests comparing mini_sklearn and sklearn RandomForestClassifier."""

    @pytest.fixture
    def classification_data(self):
        """Create a classification dataset for testing."""
        # Use make_classification for a realistic dataset
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def simple_data(self):
        """Create simple 2D classification data."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        # Create binary classification based on simple rule
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_basic_functionality(self, simple_data):
        """Test that RandomForest can fit and predict without errors."""
        X, y = simple_data

        # Our implementation
        rf_mini = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_mini.fit(X, y)
        predictions = rf_mini.predict(X)

        # Basic checks
        assert hasattr(rf_mini, 'estimators_')
        assert len(rf_mini.estimators_) == 10
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)

    def test_reproducibility(self, simple_data):
        """Test that same random_state produces same results."""
        X, y = simple_data

        # First fit
        rf1 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf1.fit(X, y)
        pred1 = rf1.predict(X)

        # Second fit with same random_state
        rf2 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf2.fit(X, y)
        pred2 = rf2.predict(X)

        # Should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_performance_reasonable(self, classification_data):
        """Test that our RandomForest achieves reasonable performance."""
        X, y = classification_data

        # Split data
        from mini_sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Our implementation
        rf_mini = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_mini.fit(X_train, y_train)

        # Check training accuracy (should be high due to overfitting tendency)
        train_score = rf_mini.score(X_train, y_train)
        assert train_score >= 0.7, f"Training accuracy too low: {train_score}"

        # Check test accuracy (should be reasonable)
        test_score = rf_mini.score(X_test, y_test)
        assert test_score >= 0.6, f"Test accuracy too low: {test_score}"

    def test_accuracy_vs_sklearn_tolerance(self, classification_data):
        """Test that our accuracy is within tolerance of sklearn's."""
        X, y = classification_data

        # Split data using the same method for both implementations
        from mini_sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Our implementation
        rf_mini = RandomForestClassifier(
            n_estimators=20,
            max_depth=10,
            random_state=42
        )
        rf_mini.fit(X_train, y_train)
        score_mini = rf_mini.score(X_test, y_test)

        # sklearn implementation
        rf_sklearn = SklearnRandomForest(
            n_estimators=20,
            max_depth=10,
            random_state=42
        )
        rf_sklearn.fit(X_train, y_train)
        score_sklearn = rf_sklearn.score(X_test, y_test)

        # Our accuracy should be within 0.25 of sklearn's
        # (allowing for differences in tree-building algorithms - our trees use random splits)
        accuracy_diff = abs(score_mini - score_sklearn)
        assert accuracy_diff <= 0.25, (
            f"Accuracy difference too large: mini={score_mini:.3f}, "
            f"sklearn={score_sklearn:.3f}, diff={accuracy_diff:.3f}"
        )

    def test_overfitting_pattern(self, classification_data):
        """Test that we see expected overfitting patterns."""
        X, y = classification_data

        # Split data
        from mini_sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(X_train, y_train)

        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)

        # Training score should be >= test score (overfitting tendency)
        assert train_score >= test_score, (
            f"Expected train_score >= test_score, got train={train_score:.3f}, "
            f"test={test_score:.3f}"
        )

    def test_different_n_estimators(self, simple_data):
        """Test that different n_estimators values work."""
        X, y = simple_data

        for n_est in [1, 5, 10, 20]:
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf.fit(X, y)
            predictions = rf.predict(X)

            assert len(rf.estimators_) == n_est
            assert len(predictions) == len(y)

    def test_predict_proba_functionality(self, simple_data):
        """Test that predict_proba works and returns valid probabilities."""
        X, y = simple_data

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        probas = rf.predict_proba(X)

        # Check shape
        assert probas.shape == (len(X), len(np.unique(y)))

        # Check probabilities sum to 1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

        # Check probabilities are between 0 and 1
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_error_handling(self, simple_data):
        """Test error handling for invalid parameters."""
        X, y = simple_data

        # Invalid n_estimators
        with pytest.raises(ValueError, match="n_estimators should be > 0"):
            rf = RandomForestClassifier(n_estimators=0)
            rf.fit(X, y)

        with pytest.raises(ValueError, match="n_estimators should be > 0"):
            rf = RandomForestClassifier(n_estimators=-1)
            rf.fit(X, y)

        # Invalid min_samples_split
        with pytest.raises(ValueError, match="min_samples_split should be >= 2"):
            rf = RandomForestClassifier(min_samples_split=1)
            rf.fit(X, y)

        # Predict before fit
        rf = RandomForestClassifier()
        with pytest.raises(ValueError, match="not fitted yet"):
            rf.predict(X)

        with pytest.raises(ValueError, match="not fitted yet"):
            rf.predict_proba(X)

    def test_max_features_options(self, simple_data):
        """Test different max_features options."""
        X, y = simple_data
        n_features = X.shape[1]

        # Test "sqrt"
        rf_sqrt = RandomForestClassifier(
            n_estimators=5, max_features="sqrt", random_state=42
        )
        rf_sqrt.fit(X, y)
        predictions_sqrt = rf_sqrt.predict(X)
        assert len(predictions_sqrt) == len(y)

        # Test integer value
        rf_int = RandomForestClassifier(
            n_estimators=5, max_features=2, random_state=42
        )
        rf_int.fit(X, y)
        predictions_int = rf_int.predict(X)
        assert len(predictions_int) == len(y)

        # Test None (all features)
        rf_none = RandomForestClassifier(
            n_estimators=5, max_features=None, random_state=42
        )
        rf_none.fit(X, y)
        predictions_none = rf_none.predict(X)
        assert len(predictions_none) == len(y)

    def test_bootstrap_vs_no_bootstrap(self, simple_data):
        """Test bootstrap vs no bootstrap sampling."""
        X, y = simple_data

        # With bootstrap
        rf_bootstrap = RandomForestClassifier(
            n_estimators=10, bootstrap=True, random_state=42
        )
        rf_bootstrap.fit(X, y)
        pred_bootstrap = rf_bootstrap.predict(X)

        # Without bootstrap
        rf_no_bootstrap = RandomForestClassifier(
            n_estimators=10, bootstrap=False, random_state=42
        )
        rf_no_bootstrap.fit(X, y)
        pred_no_bootstrap = rf_no_bootstrap.predict(X)

        # Both should work and produce valid predictions
        assert len(pred_bootstrap) == len(y)
        assert len(pred_no_bootstrap) == len(y)
        assert all(pred in [0, 1] for pred in pred_bootstrap)
        assert all(pred in [0, 1] for pred in pred_no_bootstrap)

    def test_classes_attribute(self, simple_data):
        """Test that classes_ attribute is set correctly."""
        X, y = simple_data

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)

        expected_classes = np.unique(y)
        np.testing.assert_array_equal(rf.classes_, expected_classes)
        assert rf.n_classes_ == len(expected_classes)

    def test_multiclass_classification(self):
        """Test that RandomForest works with multiclass problems."""
        # Create 3-class problem
        np.random.seed(42)
        X = np.random.randn(150, 4)
        y = np.random.choice([0, 1, 2], size=150)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset(set([0, 1, 2]))
        assert rf.n_classes_ == 3

        # Test predict_proba
        probas = rf.predict_proba(X)
        assert probas.shape == (len(X), 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)