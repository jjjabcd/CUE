from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from cue.ltau.descriptor import (
    load_error_trajectories_csv,
    make_shared_bin_edges,
    trajectories_to_pdfs,
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
) -> Tuple[LTAUEstimator, pd.DataFrame]:
    """
    Args:
        train_csv (str): Path to train_trajectories.csv
        train_descriptors (np.ndarray): (N_train, D)
        num_bins (int)
        k (int)
        save_dir (Optional[str])
        emax_scale (float)

    Returns:
        Tuple[LTAUEstimator, pd.DataFrame]:
            - fitted estimator
            - df_train: columns ['compound_chembl_id', 'uncertainty']
    """
    compound_ids, loss_trajectories = load_error_trajectories_csv(train_csv)

    if train_descriptors.shape[0] != len(compound_ids):
        raise ValueError(
            f"train_descriptors N={train_descriptors.shape[0]} != loss_trajectories N={len(compound_ids)}"
        )

    shared_bin_edges = make_shared_bin_edges(loss_trajectories, num_bins=num_bins, emax_scale=emax_scale)
    art = trajectories_to_pdfs(compound_ids, loss_trajectories, shared_bin_edges)

    u = pdfs_to_uncertainty_entropy(art.pdfs, art.shared_bin_edges)

    df_train = pd.DataFrame({
        "compound_chembl_id": art.compound_ids,
        "uncertainty": u.astype(float),
    })

    est = LTAUEstimator(k=k)
    est.fit(ref_emb=train_descriptors, reference_pdfs=art.pdfs, shared_bin_edges=art.shared_bin_edges)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df_train.to_csv(os.path.join(save_dir, "train_uncertainty.csv"), index=False)

        np.save(os.path.join(save_dir, "train_pdfs.npy"), art.pdfs)
        np.save(os.path.join(save_dir, "train_bin_edges.npy"), art.shared_bin_edges)
        np.save(os.path.join(save_dir, "train_descriptors.npy"), train_descriptors)

    return est, df_train


def estimate_uncertainty_for_split(
    estimator: LTAUEstimator,
    split_ids: np.ndarray,
    split_descriptors: np.ndarray,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Args:
        estimator (LTAUEstimator)
        split_ids (np.ndarray): (N,)
        split_descriptors (np.ndarray): (N, D)
        save_path (Optional[str])

    Returns:
        pd.DataFrame: columns ['compound_chembl_id', 'uncertainty']
    """
    u = estimator.predict_uncertainty(split_descriptors)

    df = pd.DataFrame({
        "compound_chembl_id": split_ids,
        "uncertainty": u.astype(float),
    })

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
