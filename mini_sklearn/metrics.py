"""
Metrics for evaluating model performance.

This module provides functions to compute various performance metrics
for classification and regression tasks.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    score : float
        Accuracy score (fraction of correctly classified samples)

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    if len(y_true) == 0:
        return 0.0

    return np.mean(y_true == y_pred)