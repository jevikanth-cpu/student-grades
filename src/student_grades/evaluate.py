"""Metrics for regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE and MAE.

    Args:
        y_true: True grades.
        y_pred: Predicted grades.

    Returns:
        Dictionary with ``rmse`` and ``mae``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred))}
