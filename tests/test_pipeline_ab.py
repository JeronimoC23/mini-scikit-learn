"""
A/B tests for Pipeline functionality against scikit-learn.

This module compares our implementation of Pipeline with
scikit-learn's implementation to ensure functional parity.
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from mini_sklearn.pipeline import Pipeline
from mini_sklearn.preprocessing import MinMaxScaler
from mini_sklearn.ensemble import RandomForestClassifier
from mini_sklearn.model_selection import train_test_split


class TestPipelineAB:
    """A/B tests comparing mini_sklearn and sklearn Pipeline."""

    @pytest.fixture
    def classification_data(self):
        """
        Create classification dataset matching the tutorial.

        Returns two-step split: train / (val+test), then val / test
        to match the tutorial's approach.
        """
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        # Create synthetic data
        X = np.random.uniform(-9, 9, size=(n_samples, n_features))
        # Create binary classification target
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        return X, y

    @pytest.fixture
    def split_data(self, classification_data):
        """
        Split data into train/val/test following the tutorial's two-step approach.
        """
        X, y = classification_data

        # First split: train vs rest (seed A)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

        # Second split: val vs test (seed B)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=123, stratify=y_rest
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def test_basic_pipeline_structure(self):
        """Test that Pipeline accepts valid steps and rejects invalid ones."""
        # Valid pipeline
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', RandomForestClassifier())
        ])
        assert len(pipe) == 2
        assert 'scaler' in pipe.named_steps
        assert 'clf' in pipe.named_steps

        # Test __getitem__ by name
        assert isinstance(pipe['scaler'], MinMaxScaler)
        assert isinstance(pipe['clf'], RandomForestClassifier)

        # Test __getitem__ by index
        assert isinstance(pipe[0], MinMaxScaler)
        assert isinstance(pipe[1], RandomForestClassifier)

    def test_pipeline_validation_errors(self):
        """Test that Pipeline validates steps correctly."""
        # Empty steps
        with pytest.raises(ValueError, match="Pipeline requires at least one step"):
            Pipeline([])

        # Not a list
        with pytest.raises(TypeError, match="steps should be a list"):
            Pipeline("not a list")

        # Invalid step format
        with pytest.raises(TypeError, match="Each step should be a tuple"):
            Pipeline([MinMaxScaler()])

        # Intermediate step without transform
        with pytest.raises(TypeError, match="All intermediate steps should have 'fit' and 'transform'"):
            Pipeline([
                ('bad_step', RandomForestClassifier()),  # No transform method
                ('clf', RandomForestClassifier())
            ])

        # Final step without predict
        with pytest.raises(TypeError, match="Last step should have 'fit' and 'predict'"):
            Pipeline([
                ('scaler', MinMaxScaler())  # No predict method
            ])

    def test_pipeline_fit_predict_score(self, split_data):
        """Test basic pipeline fit, predict, and score functionality."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Create and fit pipeline
        pipe = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])

        # Fit
        pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)

        # Check prediction properties
        assert y_pred.shape == (len(y_test),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        # Score
        score = pipe.score(X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_pipeline_vs_sklearn_accuracy_test_set(self, split_data):
        """
        Test that pipeline accuracy on test set is within tolerance of sklearn.

        This is the main A/B test matching the tutorial's requirements.
        Tolerance: |Δ accuracy| ≤ 0.20

        Note: Since the pipeline uses RandomForestClassifier, which has inherent
        variance due to different tree-building algorithms, we use a tolerance
        consistent with the RandomForest A/B tests (0.25). For pipelines, 0.20
        is a reasonable target.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Our implementation
        pipe_mini = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_mini.fit(X_train, y_train)
        score_mini = pipe_mini.score(X_test, y_test)

        # sklearn implementation
        pipe_sk = SklearnPipeline([
            ('scaler', SklearnMinMaxScaler(feature_range=(-1, 1))),
            ('clf', SklearnRandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_sk.fit(X_train, y_train)
        score_sk = pipe_sk.score(X_test, y_test)

        # Assert within tolerance
        diff = abs(score_mini - score_sk)
        assert diff <= 0.20, (
            f"Accuracy difference too large: {diff:.3f} "
            f"(mini: {score_mini:.3f}, sklearn: {score_sk:.3f})"
        )

    def test_pipeline_vs_sklearn_predictions_shape(self, split_data):
        """Test that predictions have correct shape and dtype."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Our implementation
        pipe_mini = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe_mini.fit(X_train, y_train)
        y_pred_mini = pipe_mini.predict(X_test)

        # sklearn implementation
        pipe_sk = SklearnPipeline([
            ('scaler', SklearnMinMaxScaler(feature_range=(-1, 1))),
            ('clf', SklearnRandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe_sk.fit(X_train, y_train)
        y_pred_sk = pipe_sk.predict(X_test)

        # Shape should match
        assert y_pred_mini.shape == y_pred_sk.shape
        assert y_pred_mini.shape == (len(y_test),)

        # Values should be in {0, 1}
        assert set(np.unique(y_pred_mini)).issubset({0, 1})

    def test_pipeline_no_data_leakage(self, split_data):
        """
        Test that pipeline doesn't leak data during prediction.

        This ensures that transform (not fit_transform) is called during
        predict and score on new data.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Create pipeline
        scaler = MinMaxScaler(feature_range=(-1, 1))
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        pipe = Pipeline([
            ('scaler', scaler),
            ('clf', clf)
        ])

        # Fit on training data
        pipe.fit(X_train, y_train)

        # Get the scaler's learned parameters
        data_min_after_fit = scaler.data_min_.copy()
        data_max_after_fit = scaler.data_max_.copy()

        # Predict on test data (should use transform, not fit_transform)
        y_pred = pipe.predict(X_test)

        # Scaler parameters should not have changed
        np.testing.assert_array_equal(scaler.data_min_, data_min_after_fit)
        np.testing.assert_array_equal(scaler.data_max_, data_max_after_fit)

        # Score on test data
        score = pipe.score(X_test, y_test)

        # Scaler parameters should still not have changed
        np.testing.assert_array_equal(scaler.data_min_, data_min_after_fit)
        np.testing.assert_array_equal(scaler.data_max_, data_max_after_fit)

    def test_pipeline_train_val_test_workflow(self, split_data):
        """
        Test complete train/val/test workflow matching the tutorial.

        Verifies:
        - Training accuracy >= validation accuracy (overfitting check)
        - Test accuracy is reasonable
        - All scores are within valid range [0, 1]
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Create pipeline
        pipe = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Fit on training data
        pipe.fit(X_train, y_train)

        # Evaluate on all three sets
        score_train = pipe.score(X_train, y_train)
        score_val = pipe.score(X_val, y_val)
        score_test = pipe.score(X_test, y_test)

        # All scores should be in valid range
        assert 0.0 <= score_train <= 1.0
        assert 0.0 <= score_val <= 1.0
        assert 0.0 <= score_test <= 1.0

        # Training accuracy should be >= validation accuracy (reasonable overfitting)
        # Note: This is a soft check, not always guaranteed but generally expected
        # for random forests with sufficient trees
        assert score_train >= score_val - 0.05, (
            f"Training accuracy ({score_train:.3f}) should be >= "
            f"validation accuracy ({score_val:.3f})"
        )

    def test_pipeline_multiple_transformers(self, split_data):
        """Test pipeline with multiple transformer steps."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Pipeline with two scalers (contrived but valid)
        pipe = Pipeline([
            ('scaler1', MinMaxScaler(feature_range=(0, 1))),
            ('scaler2', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

        assert 0.0 <= score <= 1.0
        assert len(pipe) == 3

    def test_pipeline_single_estimator(self, split_data):
        """Test pipeline with only a single estimator (no transformers)."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Pipeline with just classifier
        pipe = Pipeline([
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)

        assert 0.0 <= score <= 1.0
        assert y_pred.shape == (len(y_test),)

    def test_pipeline_reproducibility(self, split_data):
        """Test that pipelines with same random_state produce identical results."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # First pipeline
        pipe1 = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe1.fit(X_train, y_train)
        y_pred1 = pipe1.predict(X_test)
        score1 = pipe1.score(X_test, y_test)

        # Second pipeline with same seed
        pipe2 = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe2.fit(X_train, y_train)
        y_pred2 = pipe2.predict(X_test)
        score2 = pipe2.score(X_test, y_test)

        # Should be identical
        np.testing.assert_array_equal(y_pred1, y_pred2)
        assert score1 == score2

    def test_pipeline_vs_manual_chaining(self, split_data):
        """
        Test that pipeline produces same results as manual chaining.

        This verifies that the pipeline correctly chains fit_transform and transform.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Using Pipeline
        pipe = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        y_pred_pipe = pipe.predict(X_test)

        # Manual chaining
        scaler_manual = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler_manual.fit_transform(X_train)

        clf_manual = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_manual.fit(X_train_scaled, y_train)

        X_test_scaled = scaler_manual.transform(X_test)
        y_pred_manual = clf_manual.predict(X_test_scaled)

        # Should produce identical results
        np.testing.assert_array_equal(y_pred_pipe, y_pred_manual)

    def test_pipeline_error_on_score_without_method(self, split_data):
        """Test that pipeline raises error if final estimator lacks score method."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Create a dummy estimator without score method
        class DummyEstimator:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))

        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('dummy', DummyEstimator())
        ])

        pipe.fit(X_train, y_train)

        # Should raise error when trying to score
        with pytest.raises(AttributeError, match="does not implement score method"):
            pipe.score(X_test, y_test)

    def test_pipeline_comparison_summary(self, split_data):
        """
        Summary test that prints comparison results for documentation.

        This test always passes but provides useful comparison info.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data

        # Our implementation
        pipe_mini = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_mini.fit(X_train, y_train)

        score_train_mini = pipe_mini.score(X_train, y_train)
        score_val_mini = pipe_mini.score(X_val, y_val)
        score_test_mini = pipe_mini.score(X_test, y_test)

        # sklearn implementation
        pipe_sk = SklearnPipeline([
            ('scaler', SklearnMinMaxScaler(feature_range=(-1, 1))),
            ('clf', SklearnRandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_sk.fit(X_train, y_train)

        score_train_sk = pipe_sk.score(X_train, y_train)
        score_val_sk = pipe_sk.score(X_val, y_val)
        score_test_sk = pipe_sk.score(X_test, y_test)

        print("\n" + "="*60)
        print("PIPELINE A/B TEST COMPARISON")
        print("="*60)
        print(f"{'Split':<10} {'mini-sklearn':<15} {'sklearn':<15} {'Δ':<10}")
        print("-"*60)
        print(f"{'Train':<10} {score_train_mini:>14.3f} {score_train_sk:>14.3f} {abs(score_train_mini - score_train_sk):>9.3f}")
        print(f"{'Val':<10} {score_val_mini:>14.3f} {score_val_sk:>14.3f} {abs(score_val_mini - score_val_sk):>9.3f}")
        print(f"{'Test':<10} {score_test_mini:>14.3f} {score_test_sk:>14.3f} {abs(score_test_mini - score_test_sk):>9.3f}")
        print("="*60)
        print(f"Tolerance: ≤ 0.20 on test set")
        print(f"Test passes: {abs(score_test_mini - score_test_sk) <= 0.20}")
        print("="*60)
