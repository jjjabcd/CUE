from __future__ import annotations

from typing import Optional
import numpy as np


class UncertaintyNormalizer:
    """
    Args:
        mode (str): 'none' | 'minmax' | 'zscore'
    """

    def __init__(self, mode: str = "none"):
        self.mode = mode

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """
        Args:
            u (np.ndarray): Uncertainty matrix, shape (N, M)

        Returns:
            np.ndarray: Normalized uncertainty matrix, shape (N, M)
        """
        if self.mode == "none":
            return u

        if self.mode == "minmax":
            u_min = u.min(axis=0, keepdims=True)
            u_max = u.max(axis=0, keepdims=True)
            denom = np.clip(u_max - u_min, 1e-8, None)
            return (u - u_min) / denom

        if self.mode == "zscore":
            mu = u.mean(axis=0, keepdims=True)
            sd = np.clip(u.std(axis=0, keepdims=True), 1e-8, None)
            return (u - mu) / sd

        raise ValueError(f"Unknown uncertainty_norm: {self.mode}")


def compute_fusion_weights(
    u: np.ndarray,
    policy: str = "softmax",
    temperature: float = 1.0,
    random_state: Optional[int] = None,
    custom_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Args:
        u (np.ndarray): Uncertainty matrix, shape (N, M)
        policy (str): 'softmax' | 'random' | 'custom'
        temperature (float): Softmax temperature, must be > 0
        random_state (Optional[int]): RNG seed for 'random' policy
        custom_weight (Optional[np.ndarray]): Fixed per-modality weight, length M,
            required for policy='custom' (e.g. [0.75, 0.25] for fp/img)

    Returns:
        np.ndarray: Weight matrix, shape (N, M), each row sums to 1
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if policy == "softmax":
        logits = -u / temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        expv = np.exp(logits)
        return expv / expv.sum(axis=1, keepdims=True)

    if policy == "uniform":
        w = np.ones_like(u)
        return w / w.sum(axis=1, keepdims=True)

    if policy == "random":
        # Sample-wise가 아니라, 학습 1회당 Dirichlet(1,...,1)에서 modality 가중치를 한 번만
        # 뽑아 모든 샘플에 동일하게 고정 적용한다. train/val/test 호출 모두 같은
        # random_state(=args.seed)를 넘기므로 같은 run 안에서는 항상 동일한 가중치가 나온다.
        rng = np.random.default_rng(random_state)
        num_modalities = u.shape[1]
        w_fixed = rng.dirichlet(np.ones(num_modalities))
        return np.tile(w_fixed, (u.shape[0], 1))

    if policy == "sum":
        # 정규화하지 않고 모든 modality에 weight=1을 그대로 부여해 임베딩을 단순 합산한다
        # (emb = e_fp + e_img). 'uniform'(0.5/0.5 평균)과 달리 합이 1로 정규화되지 않는다.
        return np.ones_like(u)

    if policy == "custom":
        # 사용자가 지정한 고정 가중치(예: fp=0.75, img=0.25)를 모든 샘플에 동일하게 적용한다.
        if custom_weight is None:
            raise ValueError("custom_weight must be provided for fusion_policy='custom'")

        w_fixed = np.asarray(custom_weight, dtype=np.float64)
        num_modalities = u.shape[1]
        if w_fixed.ndim != 1 or w_fixed.shape[0] != num_modalities:
            raise ValueError(
                f"custom_weight must be a 1D array of length {num_modalities}, got shape {w_fixed.shape}"
            )
        return np.tile(w_fixed, (u.shape[0], 1))

    raise ValueError(f"Unknown fusion_policy: {policy}")
