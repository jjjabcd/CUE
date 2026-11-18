import numpy as np
from typing import Dict, Any

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    auc,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve
)
from scipy.stats import pearsonr, spearmanr


def _get_cindex(Y, P) -> float:
    """
    Concordance Index (CI) as used in DeepDTA emetrics.py.

    Args:
        Y (np.ndarray): Ground-truth values (array-like)
        P (np.ndarray): Predicted values (array-like)

    Returns:
        float: CI score in [0, 1] (0 if no permissible pairs)
    """
    y = np.asarray(Y, dtype=float).reshape(-1)
    p = np.asarray(P, dtype=float).reshape(-1)

    summ = 0.0
    pair = 0.0

    for i in range(1, len(y)):
        for j in range(0, i):
            if y[i] > y[j]:
                pair += 1.0
                summ += 1.0 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])

    if pair != 0.0:
        return float(summ / pair)
    return 0.0


def _get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Args:
        y_true (np.ndarray): Ground truth values, shape (N,)
        y_pred (np.ndarray): Predicted values, shape (N,)

    Returns:
        Dict[str, float]: mse, rmse, r2, pcc, ci
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_hat = np.asarray(y_pred, dtype=float).reshape(-1)

    mse = mean_squared_error(y_true, y_hat)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_hat))
    pcc = float(pearsonr(y_true, y_hat)[0]) if len(y_true) > 1 else 0.0
    srcc = float(spearmanr(y_true, y_hat)[0]) if len(y_true) > 1 else 0.0
    ci = float(_get_cindex(y_true, y_hat))

    return {"mse": float(mse), "rmse": rmse, "r2": r2, "pcc": pcc, "srcc": srcc, "ci": ci}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): logits or scores

    Returns:
        np.ndarray: probabilities in (0, 1)
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Args:
        y_true (np.ndarray): Ground truth labels in {0,1}, shape (N,)
        y_pred (np.ndarray): logits or probabilities, shape (N,)
        threshold (float): decision threshold on probability

    Returns:
        Dict[str, float]: f1, precision, recall, accuracy, auroc, auprc
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    # If score looks like logits, we still accept it; always map to prob.
    y_prob = _sigmoid(y_pred)

    y_pred = (y_prob >= float(threshold)).astype(int)
    y_true_i = y_true.astype(int)

    precision = float(precision_score(y_true_i, y_pred, zero_division=0))
    recall = float(recall_score(y_true_i, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_i, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true_i, y_pred))
    precision_pts, recall_pts, _ = precision_recall_curve(y_true_i, y_prob)
    auprc = float(auc(recall_pts, precision_pts))
    # AUROC requires both classes present
    try:
        auroc = float(roc_auc_score(y_true_i, y_prob))
    except Exception:
        auroc = 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "auroc": auroc,
        "auprc": auprc
    }


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str = "regression") -> Dict[str, Any]:
    """
    Args:
        y_true (np.ndarray): Ground truth
        y_pred (np.ndarray): Predicted values (regression) or logits/prob (classification)
        task_type (str): 'regression' or 'classification'

    Returns:
        Dict[str, Any]: metrics
    """
    if task_type == "classification":
        return _get_classification_metrics(y_true=y_true, y_pred=y_pred)
    return _get_regression_metrics(y_true=y_true, y_pred=y_pred)
