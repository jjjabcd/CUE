"""
Shared runner for single-modality DownstreamMLP training, used by the
per-modality entry points at the repo root (smi_train.py, gnn_train.py,
conf_train.py). Trains cue4modal's DownstreamMLP directly on a single
precomputed modality embedding - no CUE modality-MLP pretraining, no LTAU
uncertainty, no fusion. Same DownstreamMLP layer structure/hyperparameters as
train.py's head, same fold/CSV data format; only the embedding fed into it
changes.

Embeddings must be precomputed first via:
    python -m cue.utils.prepare_features --data_dir <...> --modalities smi gnn conf
(see scripts/run_prepare_feature.sh)
"""
from __future__ import annotations

import re
import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import tqdm

from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs
from cue.core.trainer import DownstreamTrainer
from cue.data.normalization import LabelNormalizer
from cue.utils.seed import seed
from cue.utils.utils import load_fold, aggregate_metrics

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.meshgrid")

MODALITY_TO_NPY_KEY = {
    "fp": "fp_features",
    "img": "img_features",
    "smi_mean": "smi_features_mean",
    "smi_cls": "smi_features_cls",
    "gnn": "gnn_features",
    "gnn_mgssl": "gnn_features_mgssl",
    "conf": "conf_features",
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    """
    Args:
        p (argparse.ArgumentParser): parser to add the shared downstream-training
            CLI options to (data/output paths, label handling, MLP head hparams).
    """
    p.add_argument("--data_dir", type=str, required=True, help="on-disk assay folder, e.g. ../chembl/data/selectivity_df/Kd (= DIS)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--target_col", type=str, default="Ssel")
    p.add_argument("--label_mode", type=str, default="zscore")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])

    # Downstream MLP head (same knobs as train.py's head)
    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)
    p.add_argument("--reg_num_epochs", type=int, default=500)
    p.add_argument("--reg_learning_rate", type=float, default=1e-3)
    p.add_argument("--reg_batch_size", type=int, default=256)
    p.add_argument("--reg_save_every", type=int, default=1)


def load_embedding_lookup(fold_path: str, modality: str) -> Dict[str, np.ndarray]:
    """
    Args:
        fold_path (str): fold directory (contains selectivity_{split}.npy, produced
            by cue.utils.prepare_features)
        modality (str): one of MODALITY_TO_NPY_KEY

    Returns:
        Dict[str, np.ndarray]: SMILES -> embedding vector
    """
    npy_key = MODALITY_TO_NPY_KEY[modality]
    lookup: Dict[str, np.ndarray] = {}

    for split in ("train", "val", "test"):
        npy_path = os.path.join(fold_path, f"selectivity_{split}.npy")
        if not os.path.exists(npy_path):
            continue
        d = np.load(npy_path, allow_pickle=True).item()
        if npy_key not in d:
            raise KeyError(
                f"'{npy_key}' not found in {npy_path}. "
                f"Run `python -m cue.utils.prepare_features --modalities {modality.split('_')[0]}` first."
            )
        for smi, vec in zip(d["smiles"], d[npy_key]):
            lookup[smi] = np.asarray(vec, dtype=np.float32)

    if not lookup:
        raise FileNotFoundError(
            f"No cached .npy embeddings found under {fold_path}. Run scripts/run_prepare_feature.sh first."
        )
    return lookup


def embed(X: np.ndarray, lookup: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Args:
        X (np.ndarray): SMILES array, shape (N,)
        lookup (Dict[str, np.ndarray]): SMILES -> embedding

    Returns:
        np.ndarray: embeddings, shape (N, D)
    """
    return np.stack([lookup[s] for s in X]).astype(np.float32)


def run_downstream(args: argparse.Namespace, modality: str) -> None:
    """
    Args:
        args (argparse.Namespace): parsed CLI args (see add_common_args)
        modality (str): one of MODALITY_TO_NPY_KEY

    Returns:
        None
    """
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**args.__dict__, "modality": modality}, f, indent=4)

    print("=" * 80)
    print(f"[START] Downstream-only training (modality={modality})")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  task_type : {args.task_type}")
    print("=" * 80)

    all_predictions: List[pd.DataFrame] = []

    fold_dirs = sorted(
        d for d in os.listdir(args.data_dir)
        if re.fullmatch(r"fold\d+", d) and os.path.isdir(os.path.join(args.data_dir, d))
    )
    print(f"[INFO] Found folds: {fold_dirs}")

    for fold in tqdm.tqdm(fold_dirs, desc="Downstream folds", leave=True, dynamic_ncols=True):
        seed(args.seed)
        fold_path = os.path.join(args.data_dir, fold)
        fold_out = os.path.join(args.output_dir, fold)
        os.makedirs(fold_out, exist_ok=True)

        tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te = load_fold(fold_path, args.target_col)

        lookup = load_embedding_lookup(fold_path, modality)
        H_tr, H_va, H_te = embed(X_tr, lookup), embed(X_va, lookup), embed(X_te, lookup)

        # Label normalization (same convention as train.py)
        lab = LabelNormalizer("none" if args.task_type == "classification" else args.label_mode)
        lab.fit(y_tr)
        y_tr_n = lab.transform(y_tr)
        y_va_n = lab.transform(y_va)
        y_te_n = lab.transform(y_te)

        head_dir = os.path.join(fold_out, "regressors" if args.task_type == "regression" else "classifiers")
        os.makedirs(head_dir, exist_ok=True)
        lab.save(os.path.join(head_dir, "label_normalizer.pkl"))
        mlp_ckpt_path = os.path.join(head_dir, "MLP.pth")

        head_cfg = DownstreamMLPConfigs(
            num_epochs=args.reg_num_epochs,
            learning_rate=args.reg_learning_rate,
            batch_size=args.reg_batch_size,
            save_every=args.reg_save_every,
            hidden_dim_1=args.reg_hidden_dim_1,
            hidden_dim_2=args.reg_hidden_dim_2,
            hidden_dim_3=args.reg_hidden_dim_3,
            device=str(device),
            seed=args.seed,
        )

        head_model = DownstreamMLP(
            input_dim=int(H_tr.shape[1]),
            hidden_dim_1=int(head_cfg.hidden_dim_1),
            hidden_dim_2=int(head_cfg.hidden_dim_2),
            hidden_dim_3=int(head_cfg.hidden_dim_3),
            output_dim=1,
        )

        mlp_trainer = DownstreamTrainer(model=head_model, device=device)

        history = mlp_trainer.fit(
            feature_name=modality,
            X=H_tr,
            y=y_tr_n.reshape(-1, 1) if y_tr_n.ndim == 1 else y_tr_n,
            batch_size=int(head_cfg.batch_size),
            num_epochs=int(head_cfg.num_epochs),
            learning_rate=float(head_cfg.learning_rate),
            validation_data={
                "X": H_va,
                "y": y_va_n.reshape(-1, 1) if y_va_n.ndim == 1 else y_va_n,
            },
            filepath_ckpt=mlp_ckpt_path,
            save_every=int(head_cfg.save_every),
            save_trajectory=True,
            trajectory_dir=os.path.join(head_dir, "trajectories"),
            compound_ids_tr=tr_id,
            compound_ids_va=va_id,
            smiles_tr=X_tr,
            smiles_va=X_va,
            task_type=args.task_type,
            traj_error_mode="bce" if args.task_type == "classification" else "abs",
        )

        mlp_trainer.plot(
            history=history,
            save_dir=head_dir,
            task_type=args.task_type,
            best_key=("val_bce" if args.task_type == "classification" else "val_mse"),
        )

        mlp_trainer.load_model(mlp_ckpt_path)

        # Test prediction
        z_pred_te = mlp_trainer.inference(
            H_te,
            batch_size=1024,
            task_type=args.task_type,
            return_logits=True,
        )

        fold_records: List[Dict[str, Any]] = []
        for i, (cid, gt) in enumerate(zip(te_id, y_te)):
            row: Dict[str, Any] = {
                "compound_chembl_id": cid,
                "GroundTruth": float(gt),
                "Fold": fold,
            }
            if args.task_type == "regression":
                row["mlp"] = float(lab.inverse_transform(float(z_pred_te[i])))
            else:
                logit = float(z_pred_te[i])
                row["mlp"] = float(1.0 / (1.0 + np.exp(-logit)))
            fold_records.append(row)

        df_fold = pd.DataFrame(fold_records)
        df_fold.to_csv(os.path.join(fold_out, "predictions.csv"), index=False)
        all_predictions.append(df_fold)

        print(f"[{fold}] DONE")

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    print("=" * 80)
    print("[FINAL] Aggregating metrics")

    df_all = pd.concat(all_predictions, ignore_index=True)
    df_metrics = aggregate_metrics(df_all, task_type=args.task_type)
    df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    df_all.to_csv(os.path.join(args.output_dir, "predictions_all.csv"), index=False)

    print("[DONE] All folds completed")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
