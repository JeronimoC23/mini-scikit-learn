"""
Data splitting utilities for train/validation/test splits.

This module provides functions to split datasets while optionally preserving
class distributions (stratification).
"""

import numpy as np
from .base import check_X_y, _rng


def train_test_split(X, y, *, test_size=0.2, random_state=None, stratify=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split
    stratify : array-like or None, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels

    Returns
    -------
    X_train : ndarray
        Training data
    X_test : ndarray
        Test data
    y_train : ndarray
        Training targets
    y_test : ndarray
        Test targets

    Raises
    ------
    ValueError
        If test_size is not between 0 and 1, or if stratification fails
        due to insufficient samples per class
    """
    # Validate inputs
    X, y = check_X_y(X, y)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size should be between 0 and 1, got {test_size}")

    n_samples = X.shape[0]
    n_test = int(np.ceil(test_size * n_samples))
    n_train = n_samples - n_test

    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"With n_samples={n_samples} and test_size={test_size}, "
            f"the resulting train size={n_train} or test size={n_test} is 0"
        )

    rng = _rng(random_state)

    if stratify is None:
        # Simple random split
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # Stratified split
        stratify = np.asarray(stratify)
        if len(stratify) != n_samples:
            raise ValueError(
                f"stratify has {len(stratify)} samples but X has {n_samples}"
            )

        # Get unique classes and their counts
        classes, y_indices = np.unique(stratify, return_inverse=True)
        n_classes = len(classes)

        # Calculate samples per class for test set
        class_counts = np.bincount(y_indices)
        test_counts = np.round(class_counts * test_size).astype(int)

        # Ensure we have at least 1 sample per class in test if class exists
        # and at least 1 sample per class in train
        for i, (count, test_count) in enumerate(zip(class_counts, test_counts)):
            if count == 1 and test_size > 0:
                raise ValueError(
                    f"The least populated class in stratify has only 1 member, "
                    f"which is too few. The minimum number of members in any "
                    f"class cannot be less than 2."
                )
            if test_count == 0 and count > 0:
                test_counts[i] = 1
            if test_count == count and count > 1:
                test_counts[i] = count - 1

        # Check if we still have valid splits
        if test_counts.sum() > n_test:
            # Reduce test counts proportionally
            excess = test_counts.sum() - n_test
            for i in range(len(test_counts)):
                if test_counts[i] > 1 and excess > 0:
                    reduction = min(test_counts[i] - 1, excess)
                    test_counts[i] -= reduction
                    excess -= reduction

        # Collect indices for each class
        train_indices = []
        test_indices = []

        for class_idx, (class_label, test_count) in enumerate(zip(classes, test_counts)):
            # Get all indices for this class
            class_mask = y_indices == class_idx
            class_indices = np.where(class_mask)[0]

            # Shuffle indices for this class
            rng.shuffle(class_indices)

            # Split into train/test
            test_indices.extend(class_indices[:test_count])
            train_indices.extend(class_indices[test_count:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Final shuffle to mix classes
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )