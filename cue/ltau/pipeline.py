from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from cue.ltau.descriptor import (
    load_error_trajectories_csv,
    make_shared_bin_edges,
    trajectories_to_pdfs,
    pdfs_to_expected_error,
    pdfs_to_uncertainty_entropy,
)
from cue.ltau.estimator import LTAUEstimator


def fit_ltau_estimator_from_train_trajectories(
    train_csv: str,
    train_descriptors: np.ndarray,
    num_bins: int,
    k: int,
    save_dir: Optional[str] = None,
    emax_scale: float = 1.2,
    uncertainty_mode: str = "entropy",
) -> Tuple[LTAUEstimator, pd.DataFrame]:
    """
    Args:
        train_csv (str): Path to train_trajectories.csv
        train_descriptors (np.ndarray): (N_train, D)
        num_bins (int)
        k (int)
        save_dir (Optional[str])
        emax_scale (float)
        uncertainty_mode (str): 'entropy' (recommended) or 'expected_error'

    Returns:
        Tuple[LTAUEstimator, pd.DataFrame]:
            - fitted estimator
            - df_train: columns ['compound_chembl_id', 'uncertainty', 'expected_error']
    """
    compound_ids, trajectories = load_error_trajectories_csv(train_csv)

    if train_descriptors.shape[0] != len(compound_ids):
        raise ValueError(
            f"train_descriptors N={train_descriptors.shape[0]} != trajectories N={len(compound_ids)}"
        )

    bin_edges = make_shared_bin_edges(trajectories, num_bins=num_bins, emax_scale=emax_scale)
    art = trajectories_to_pdfs(compound_ids, trajectories, bin_edges)

    # Train uncertainty (directly from its own PDF)
    u_ent = pdfs_to_uncertainty_entropy(art.pdfs, art.bin_edges)
    u_exp = pdfs_to_expected_error(art.pdfs, art.bin_edges)

    if uncertainty_mode == "entropy":
        u = u_ent
    elif uncertainty_mode == "expected_error":
        u = u_exp
    else:
        raise ValueError("uncertainty_mode must be 'entropy' or 'expected_error'")

    df_train = pd.DataFrame({
        "compound_chembl_id": art.compound_ids,
        "uncertainty": u.astype(float),
        "expected_error": u_exp.astype(float),
        "uncertainty_entropy": u_ent.astype(float),
    })

    est = LTAUEstimator(k=k, use_faiss=True)
    est.fit(train_desc=train_descriptors, train_pdfs=art.pdfs, bin_edges=art.bin_edges)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df_train.to_csv(os.path.join(save_dir, "train_uncertainty.csv"), index=False)

        np.save(os.path.join(save_dir, "train_pdfs.npy"), art.pdfs)
        np.save(os.path.join(save_dir, "train_bin_edges.npy"), art.bin_edges)
        np.save(os.path.join(save_dir, "train_descriptors.npy"), train_descriptors)

    return est, df_train


def estimate_uncertainty_for_split(
    estimator: LTAUEstimator,
    split_ids: np.ndarray,
    split_descriptors: np.ndarray,
    save_path: Optional[str] = None,
    uncertainty_mode: str = "entropy",
) -> pd.DataFrame:
    """
    Args:
        estimator (LTAUEstimator)
        split_ids (np.ndarray): (N,)
        split_descriptors (np.ndarray): (N, D)
        save_path (Optional[str])
        uncertainty_mode (str): 'entropy' | 'expected_error'

    Returns:
        pd.DataFrame:
            columns ['compound_chembl_id', 'uncertainty', 'expected_error', 'uncertainty_entropy']
    """
    u_ent, _ = estimator.predict_uncertainty(split_descriptors, mode="entropy", return_pdfs=False)
    u_exp, _ = estimator.predict_uncertainty(split_descriptors, mode="expected_error", return_pdfs=False)

    if uncertainty_mode == "entropy":
        u = u_ent
    elif uncertainty_mode == "expected_error":
        u = u_exp
    else:
        raise ValueError("uncertainty_mode must be 'entropy' or 'expected_error'")

    df = pd.DataFrame({
        "compound_chembl_id": split_ids,
        "uncertainty": u.astype(float),
        "expected_error": u_exp.astype(float),
        "uncertainty_entropy": u_ent.astype(float),
    })

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
