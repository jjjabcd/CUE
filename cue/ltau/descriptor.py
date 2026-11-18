from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PDFArtifacts:
    """Container for PDF artifacts."""
    compound_ids: np.ndarray
    pdfs: np.ndarray          # (N, B) density per bin
    bin_edges: np.ndarray     # (B+1,)


def load_error_trajectories_csv(path_csv: str) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Args:
        path_csv (str): Path to *train_trajectories.csv* or *val_trajectories.csv*.

    Returns:
        Tuple[np.ndarray, List[List[float]]]:
            - compound_ids: (N,)
            - trajectories: list of length N, each is a list of errors over epochs
    """
    df = pd.read_csv(path_csv)
    if "compound_chembl_id" not in df.columns or "error_trajectory" not in df.columns:
        raise ValueError("CSV must contain columns: compound_chembl_id, error_trajectory")

    compound_ids = df["compound_chembl_id"].values
    traj_raw = df["error_trajectory"].tolist()

    trajectories: List[List[float]] = []
    for x in traj_raw:
        if isinstance(x, list):
            trajectories.append([float(v) for v in x])
        else:
            # stored as string like "[0.1, 0.2, ...]"
            trajectories.append([float(v) for v in ast.literal_eval(x)])
    return compound_ids, trajectories


def make_shared_bin_edges(
    trajectories: List[List[float]],
    num_bins: int,
    emax_scale: float = 1.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Args:
        trajectories (List[List[float]]): Error trajectories.
        num_bins (int): Number of bins.
        emax_scale (float): Scale factor for max error to define upper bin edge.
        eps (float): Small epsilon to avoid degenerate ranges.

    Returns:
        np.ndarray: Bin edges, shape (num_bins + 1,)
    """
    all_vals = np.concatenate([np.asarray(t, dtype=float) for t in trajectories])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite trajectory values found.")

    e_min = float(np.min(all_vals))
    e_max = float(np.max(all_vals))

    # Force range to start at 0 for absolute error, and avoid zero-width range
    left = 0.0
    right = max(eps, e_max * emax_scale)

    if right <= left:
        right = left + eps

    return np.linspace(left, right, num_bins + 1, dtype=float)


def trajectories_to_pdfs(
    compound_ids: np.ndarray,
    trajectories: List[List[float]],
    bin_edges: np.ndarray,
) -> PDFArtifacts:
    """
    Args:
        compound_ids (np.ndarray): (N,)
        trajectories (List[List[float]]): list length N
        bin_edges (np.ndarray): (B+1,)

    Returns:
        PDFArtifacts: pdfs are density per bin, shape (N, B)
    """
    n = len(trajectories)
    b = len(bin_edges) - 1
    pdfs = np.zeros((n, b), dtype=float)

    for i, t in enumerate(trajectories):
        arr = np.asarray(t, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        hist, _ = np.histogram(arr, bins=bin_edges, density=True)
        pdfs[i] = hist.astype(float)

    return PDFArtifacts(compound_ids=compound_ids, pdfs=pdfs, bin_edges=bin_edges)


def pdfs_to_uncertainty_entropy(pdfs: np.ndarray, bin_edges: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Args:
        pdfs (np.ndarray): Density per bin, (N, B)
        bin_edges (np.ndarray): (B+1,)
        eps (float)

    Returns:
        np.ndarray: Normalized entropy uncertainty in [0, 1], shape (N,)
    """
    widths = (bin_edges[1:] - bin_edges[:-1]).astype(float)  # (B,)
    widths = np.maximum(widths, eps)

    # Convert density -> probability mass
    pmass = pdfs * widths[None, :]
    pmass = np.maximum(pmass, 0.0)
    s = pmass.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    pmass = pmass / s

    ent = -(pmass * np.log(np.maximum(pmass, eps))).sum(axis=1)  # (N,)
    ent_max = np.log(pmass.shape[1] + eps)
    u = ent / np.maximum(ent_max, eps)
    return np.clip(u, 0.0, 1.0)


def pdfs_to_expected_error(pdfs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Args:
        pdfs (np.ndarray): Density per bin, (N, B)
        bin_edges (np.ndarray): (B+1,)

    Returns:
        np.ndarray: Expected error (same scale as trajectory), shape (N,)
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = (bin_edges[1:] - bin_edges[:-1])

    pmass = pdfs * widths[None, :]
    s = np.maximum(pmass.sum(axis=1, keepdims=True), 1e-12)
    pmass = pmass / s

    return (pmass * centers[None, :]).sum(axis=1)
