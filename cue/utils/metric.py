import numpy as np
from typing import Dict, Any

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr


def _get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)

    Returns:
        Dict[str, float]: rmse, r2, pcc
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_hat = np.asarray(y_pred, dtype=float).reshape(-1)

    mse = mean_squared_error(y_true, y_hat)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_hat))
    pcc = float(pearsonr(y_true, y_hat)[0]) if len(y_true) > 1 else 0.0

    return {"rmse": rmse, "r2": r2, "pcc": pcc}


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Args:
        y_true (np.ndarray): Ground truth
        y_pred (np.ndarray): Predicted values

    Returns:
        Dict[str, Any]: rmse, r2, pcc
    """
    return _get_regression_metrics(y_true=y_true, y_pred=y_pred)
