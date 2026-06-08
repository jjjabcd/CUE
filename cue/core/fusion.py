from __future__ import annotations

import numpy as np


def compute_fusion_weights(
    u: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Args:
        u (np.ndarray): Uncertainty matrix, shape (N, M)
        temperature (float): Softmax temperature, must be > 0

    Returns:
        np.ndarray: Weight matrix, shape (N, M), each row sums to 1
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = -u / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / expv.sum(axis=1, keepdims=True)
