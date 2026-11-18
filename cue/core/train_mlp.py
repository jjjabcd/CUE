import os
from cue.core.trainer import Trainer
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class MLPTrainConfigs:
    """
    Args:
        feature_name (str): Modality name, e.g., 'fp', 'img'
        trainer (object): Trainer instance (must implement fit/load_model)
        scaler (object): Scaler instance (must implement fit/transform)
        get_features (Callable[[str], np.ndarray]): Feature extractor function
        fold_dir (str): Fold output directory
        num_epochs (int): Training epochs
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        save_every (int): Trajectory saving frequency in epochs
        save_trajectory (bool): Whether to record per-epoch error trajectories (needed for
            LTAU; MC Dropout/Deep Ensemble don't use them, so pass False to skip the extra
            per-epoch full-dataset forward pass).
        member_idx (Optional[int]): Ensemble member index. When set and num_members > 1,
            checkpoint/history/loss/trajectory paths are suffixed with ``_m{member_idx}`` so
            multiple Deep Ensemble members don't collide.
        num_members (Optional[int]): Total number of members being trained for this modality
            (e.g. ``ensemble_size`` for Deep Ensemble). Filenames are only suffixed when this
            is > 1, so existing single-member (LTAU/MC Dropout) paths are unchanged.
    """
    feature_name: str
    trainer: object
    scaler: object
    get_features: Callable[[str], np.ndarray]
    fold_dir: str
    num_epochs: int = 2000
    learning_rate: float = 1e-4
    batch_size: int = 32
    save_every: int = 1
    task_type: str = "regression"
    save_trajectory: bool = True
    member_idx: Optional[int] = None
    num_members: Optional[int] = None

    @property
    def display_name(self) -> str:
        """
        Args:
            None

        Returns:
            str: Progress-bar label for this modality/member, e.g. "fp" for a
                single-member run or "fp [member 3/10]" for Deep Ensemble
                member index 2 (0-based) out of 10 members.
        """
        if self.member_idx is not None and self.num_members is not None and self.num_members > 1:
            return f"{self.feature_name} [member {self.member_idx + 1}/{self.num_members}]"
        return self.feature_name

    @property
    def _suffix(self) -> str:
        if self.member_idx is not None and self.num_members is not None and self.num_members > 1:
            return f"_m{self.member_idx}"
        return ""

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.fold_dir, "checkpoints")

    @property
    def traj_dir(self) -> str:
        return os.path.join(self.fold_dir, "trajectories", f"{self.feature_name}{self._suffix}")

    @property
    def filepath_ckpt(self) -> str:
        return os.path.join(self.ckpt_dir, f"mlp_{self.feature_name}{self._suffix}.pth")

    @property
    def filepath_scaler(self) -> str:
        return os.path.join(self.ckpt_dir, f"scaler_{self.feature_name}.pkl")

    @property
    def filepath_hist(self) -> str:
        return os.path.join(self.fold_dir, f"history_{self.feature_name}{self._suffix}.csv")

    @property
    def filepath_loss(self) -> str:
        return os.path.join(self.fold_dir, f"loss_{self.feature_name}{self._suffix}.png")

def history_to_df(history: Dict[str, List[float]], task_type: str) -> pd.DataFrame:
    """
    Args:
        history (Dict[str, List[float]]): History returned by Trainer.fit()
        task_type (str): 'regression' | 'classification'

    Returns:
        pd.DataFrame: Task-specific history dataframe
    """
    if task_type not in {"regression", "classification"}:
        raise ValueError("task_type must be 'regression' or 'classification'")

    base_cols = ["epoch", "loss"]

    if task_type == "classification":
        keep_cols = base_cols + [
            "val_bce",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_auc_roc",
            "val_mcc",
        ]
    else:
        keep_cols = base_cols + [
            "val_mse",
            "val_rmse",
            "val_r2",
            "val_pcc",
            "val_ci",
        ]

    n = len(history.get("epoch", []))
    out: Dict[str, List[float]] = {}

    for c in keep_cols:
        if c in history and len(history[c]) == n:
            out[c] = history[c]
        else:
            out[c] = [float("nan")] * n

    return pd.DataFrame(out)


def train_mlp(
    tr_id: np.ndarray,
    va_id: np.ndarray,
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    cfg: MLPTrainConfigs,
) -> float:
    """
    Args:
        tr_id (np.ndarray): Train compound IDs, shape (N_tr,)
        va_id (np.ndarray): Val compound IDs, shape (N_va,)
        X_tr (np.ndarray): Train SMILES, shape (N_tr,)
        X_va (np.ndarray): Val SMILES, shape (N_va,)
        y_tr (np.ndarray): Train labels (UNNORMALIZED), shape (N_tr, 1)
        y_va (np.ndarray): Val labels (UNNORMALIZED), shape (N_va, 1)
        cfg (MLPTrainConfigs): Training configuration

    Returns:
        float: Best validation RMSE (based on Trainer.evaluate outputs)
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.save_trajectory:
        os.makedirs(cfg.traj_dir, exist_ok=True)

    # -------------------------
    # Feature extraction
    # -------------------------
    inps_tr = np.array([cfg.get_features(s) for s in X_tr], dtype=np.float32)
    inps_va = np.array([cfg.get_features(s) for s in X_va], dtype=np.float32)

    # -------------------------
    # Scaling
    # -------------------------
    cfg.scaler.fit(inps_tr)
    inps_tr = cfg.scaler.transform(inps_tr)
    inps_va = cfg.scaler.transform(inps_va)
    joblib.dump(cfg.scaler, cfg.filepath_scaler)

    # -------------------------
    # Fit (Trainer.fit handles trajectory saving)
    # -------------------------
    history = cfg.trainer.fit(
        cfg.display_name,
        X=inps_tr,
        y=y_tr,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        validation_data={"X": inps_va, "y": y_va},
        filepath_ckpt=cfg.filepath_ckpt,
        save_trajectory=cfg.save_trajectory,
        trajectory_dir=cfg.traj_dir,
        save_every=cfg.save_every,
        compound_ids_tr=tr_id,
        compound_ids_va=va_id,
        smiles_tr=X_tr,
        smiles_va=X_va,
        task_type=cfg.task_type,
    )

    # -------------------------
    # Save history
    # -------------------------
    df_hist = history_to_df(history, cfg.task_type)
    df_hist.to_csv(cfg.filepath_hist, index=False)

    # -------------------------
    # Plot (matplotlib only)
    # -------------------------
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)

    ax.plot(df_hist["epoch"], df_hist["loss"])

    if cfg.task_type == "classification":
        # Validation curve: val_bce
        ax.plot(df_hist["epoch"], df_hist["val_bce"])
        best_epoch = int(np.nanargmin(df_hist["val_bce"].to_numpy(dtype=float)))
        ax.axvline(best_epoch, linestyle="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(["BCE(tr)", "BCE(va)"])
    else:
        # Validation curve: val_mse
        ax.plot(df_hist["epoch"], df_hist["val_mse"])
        best_epoch = int(np.nanargmin(df_hist["val_mse"].to_numpy(dtype=float)))
        ax.axvline(best_epoch, linestyle="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(["MSE(tr)", "MSE(va)"])

    fig.tight_layout()
    fig.savefig(cfg.filepath_loss)
    plt.close(fig)

    # Restore best checkpoint
    cfg.trainer.load_model(cfg.filepath_ckpt)

    # Return best score (used by CUE.fit)
    if cfg.task_type == "classification":
        return float(np.nanmin(df_hist["val_bce"].to_numpy(dtype=float)))
    return float(np.nanmin(df_hist["val_mse"].to_numpy(dtype=float)))
