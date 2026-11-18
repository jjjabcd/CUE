"""
Deep Evidential Regression (Amini et al., NeurIPS 2020) for CUE.

The modality MLP predicts the four parameters of a Normal-Inverse-Gamma
(NIG) distribution instead of a single point estimate::

    gamma  -- predictive mean
    nu     -- virtual observations supporting gamma
    alpha  -- virtual observations supporting the variance
    beta   -- sum of squared deviations

From these, two uncertainty terms are available::

    aleatoric  = beta / (alpha - 1)          data noise
    epistemic  = beta / (nu * (alpha - 1))   model ignorance

CUE fuses modalities by down-weighting the uncertain one, and both MC
Dropout and Deep Ensemble supply an epistemic quantity there (variance
across stochastic passes / across members). ``epistemic`` is therefore the
default so all three methods weight on the same kind of quantity; see
``EVIDENTIAL_UNCERTAINTY_TERMS`` for the alternatives.

This mirrors ``thesis/src/evidential_regression.py``, adapted to CUE's
SMILES-driven feature pipeline and uncertainty-aware fusion.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .mlp import MLP


EVIDENTIAL_UNCERTAINTY_TERMS = ("epistemic", "aleatoric", "total")


class EvidentialMLP(MLP):
    """
    The CUE modality MLP with a four-parameter NIG output head.

    Layer sizes are inherited from ``MLP`` unchanged, so the hidden
    embedding handed to fusion is identical in shape and meaning to the one
    MC Dropout and Deep Ensemble produce. Only the final layer differs: it
    emits four numbers per sample instead of one.

    ``forward`` returns gamma alone, matching the ``(N, 1)`` contract that
    ``Trainer.evaluate``/``inference`` rely on -- gamma *is* the point
    prediction under the NIG model. Training and uncertainty inference call
    ``forward_nig`` for the full parameter tuple.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.0):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Last hidden embedding dimension.
            dropout_rate (float): Dropout probability after each hidden
                ReLU. Evidential Regression draws its uncertainty from the
                predicted NIG parameters rather than from stochastic
                passes, so this defaults to 0.0.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            dropout_rate=dropout_rate,
        )

    def forward_nig(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            Tuple[torch.Tensor, ...]: ``(gamma, nu, alpha, beta)``, each of
                shape (N, 1). Softplus keeps nu and beta positive, and
                alpha is shifted past 1 so the NIG moments stay finite.
        """
        raw = super().forward(x)
        raw_gamma, raw_nu, raw_alpha, raw_beta = raw.split(1, dim=-1)
        nu = F.softplus(raw_nu)
        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta)
        return raw_gamma, nu, alpha, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            torch.Tensor: Predictive mean gamma, shape (N, 1). Returning
                gamma alone lets the inherited evaluate/inference paths run
                against this model untouched.
        """
        return self.forward_nig(x)[0]


def _log_gamma(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Lanczos approximation to log Gamma.

    ``torch.lgamma`` has no sm_89 kernel in the legacy torch build this
    repository pins, so the series is evaluated directly. It is accurate
    and numerically stable over the alpha > 1 range the NIG head produces.

    Args:
        x (torch.Tensor): Positive input tensor.

    Returns:
        torch.Tensor: Elementwise log Gamma of ``x``.
    """
    coefficients = x.new_tensor([
        676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
    ])
    z = x - 1.0
    series = x.new_tensor(0.99999999999980993)
    for index, coefficient in enumerate(coefficients, start=1):
        series = series + coefficient / (z + float(index))
    g = 7.0
    t = z + g + 0.5
    return 0.5 * x.new_tensor(np.log(2.0 * np.pi)) + (z + 0.5) * torch.log(t) - t + torch.log(series)


def nig_nll(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of the NIG evidential distribution.

    Args:
        y (torch.Tensor): Targets, shape (N, 1).
        gamma (torch.Tensor): Predictive mean, shape (N, 1).
        nu (torch.Tensor): Virtual observations for the mean, shape (N, 1).
        alpha (torch.Tensor): Virtual observations for the variance, shape (N, 1).
        beta (torch.Tensor): Sum of squared deviations, shape (N, 1).

    Returns:
        torch.Tensor: Per-sample NLL, shape (N, 1).
    """
    omega = 2.0 * beta * (1.0 + nu)
    return (
        0.5 * torch.log(torch.as_tensor(np.pi, dtype=y.dtype, device=y.device) / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma).square() * nu + omega)
        + _log_gamma(alpha) - _log_gamma(alpha + 0.5)
    )


def nig_reg(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Evidence regularizer penalizing confidence on badly-fit samples.

    Args:
        y (torch.Tensor): Targets, shape (N, 1).
        gamma (torch.Tensor): Predictive mean, shape (N, 1).
        nu (torch.Tensor): Virtual observations for the mean, shape (N, 1).
        alpha (torch.Tensor): Virtual observations for the variance, shape (N, 1).

    Returns:
        torch.Tensor: Per-sample penalty, shape (N, 1).
    """
    return torch.abs(y - gamma) * (2.0 * nu + alpha)


def evidential_loss(
    y: torch.Tensor,
    params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lambda_reg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        y (torch.Tensor): Targets, shape (N, 1).
        params (Tuple[torch.Tensor, ...]): ``(gamma, nu, alpha, beta)``.
        lambda_reg (float): Weight on the evidence regularizer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, NLL
            term, and regularizer term, each a scalar.
    """
    gamma, nu, alpha, beta = params
    nll = nig_nll(y, gamma, nu, alpha, beta).mean()
    reg = nig_reg(y, gamma, nu, alpha).mean()
    return nll + lambda_reg * reg, nll, reg


def nig_uncertainty(
    nu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    term: str = "epistemic",
) -> np.ndarray:
    """
    Reduce NIG parameters to the scalar u(x) that drives fusion weights.

    Args:
        nu (np.ndarray): Virtual observations for the mean, shape (N,).
        alpha (np.ndarray): Virtual observations for the variance, shape (N,).
        beta (np.ndarray): Sum of squared deviations, shape (N,).
        term (str): One of ``EVIDENTIAL_UNCERTAINTY_TERMS``. ``epistemic``
            is ``beta / (nu * (alpha - 1))``, the model-ignorance term that
            matches what MC Dropout and Deep Ensemble measure.

    Returns:
        np.ndarray: Non-negative uncertainty, shape (N,).

    Raises:
        ValueError: If ``term`` is not a supported uncertainty term.
    """
    if term not in EVIDENTIAL_UNCERTAINTY_TERMS:
        raise ValueError(
            f"Unsupported evidential uncertainty term={term!r}; "
            f"expected one of {EVIDENTIAL_UNCERTAINTY_TERMS}"
        )
    aleatoric = beta / (alpha - 1.0)
    if term == "aleatoric":
        u = aleatoric
    elif term == "epistemic":
        u = aleatoric / nu
    else:  # total
        u = aleatoric + aleatoric / nu
    return np.clip(u, a_min=0.0, a_max=None)
