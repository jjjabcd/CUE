from typing import List, Tuple, Dict, Any
import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from cue.utils.metric import get_metrics
import time
import pickle
import torch
from torch.utils.data import Dataset as TorchDataset

def ensure_dirs(fold_out: str) -> None:
    """
    Args:
        fold_out (str): fold output directory
    Returns:
        None
    """
    os.makedirs(fold_out, exist_ok=True)
    os.makedirs(os.path.join(fold_out, "fusion"), exist_ok=True)
    os.makedirs(os.path.join(fold_out, "emb"), exist_ok=True)
    os.makedirs(os.path.join(fold_out, "regressors"), exist_ok=True)
    os.makedirs(os.path.join(fold_out, "ltau"), exist_ok=True)




def _drop_nan_rows(
    df: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop rows with NaN in SMILES or target column.

    Args:
        df (pd.DataFrame): input dataframe
        id_col (str): compound id column name
        smiles_col (str): SMILES column name
        target_col (str): target column name

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            ids, smiles, targets (NaN-free)
    """
    before = len(df)

    df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)

    after = len(df)
    if before != after:
        print(f"[load_fold] Dropped {before - after} rows due to NaN in {smiles_col}/{target_col}")

    ids = df[id_col].values
    smiles = df[smiles_col].values
    targets = df[target_col].values.astype(float)

    return ids, smiles, targets

def load_fold(fold_path: str, target_col: str) -> Tuple[np.ndarray, ...]:
    """
    Args:
        fold_path (str): fold directory path
        target_col (str): target column name

    Returns:
        Tuple[np.ndarray, ...]:
            tr_id, va_id, te_id,
            X_tr, y_tr,
            X_va, y_va,
            X_te, y_te
    """
    tr = pd.read_csv(os.path.join(fold_path, "selectivity_train.csv"))
    va = pd.read_csv(os.path.join(fold_path, "selectivity_val.csv"))
    te = pd.read_csv(os.path.join(fold_path, "selectivity_test.csv"))

    tr_id, X_tr, y_tr = _drop_nan_rows(
        tr,
        id_col="compound_chembl_id",
        smiles_col="SMILES",
        target_col=target_col,
    )
    va_id, X_va, y_va = _drop_nan_rows(
        va,
        id_col="compound_chembl_id",
        smiles_col="SMILES",
        target_col=target_col,
    )
    te_id, X_te, y_te = _drop_nan_rows(
        te,
        id_col="compound_chembl_id",
        smiles_col="SMILES",
        target_col=target_col,
    )

    return tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te

def aggregate_metrics(df_ys: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Args:
        df_ys (pd.DataFrame): concatenated predictions across folds

    Returns:
        pd.DataFrame: aggregated metrics table (mean/std over folds)
    """
    meta_cols = {
        "GroundTruth",
        "Fold",
        "compound_chembl_id",
        "ID",
        "Dataset",
    }

    folds = sorted(df_ys["Fold"].unique())

    # regression: keep numeric columns only
    pred_cols = [c for c in df_ys.columns if c not in meta_cols]
    model_names = [c for c in pred_cols if is_numeric_dtype(df_ys[c])]

    results: List[Dict[str, Any]] = []

    for model_name in model_names:
        fold_metrics: List[Dict[str, float]] = []

        for fold in folds:
            fd = df_ys[df_ys["Fold"] == fold]

            y_true = fd["GroundTruth"].values.astype(float)
            y_pred = fd[model_name].values.astype(float)

            # --------------------------------------------------
            # NaN/Inf guard (critical for concat with missing cols)
            # --------------------------------------------------
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_f = y_true[mask]
            y_pred_f = y_pred[mask]

            # If nothing left, push zeros to keep table shape stable
            if y_true_f.size == 0:
                fold_metrics.append({})
                continue

            met = get_metrics(y_true_f, y_pred_f)
            fold_metrics.append(met)

        keys = ["rmse", "r2", "pcc"]

        row: Dict[str, Any] = {"model_name": model_name}
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in fold_metrics]
            row[k] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals))

        results.append(row)

    return pd.DataFrame(results)



def compute_fold_metrics_table(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_pred (pd.DataFrame): Fold-level prediction table (e.g., predictions.csv)

    Returns:
        pd.DataFrame: rows = models, columns = metrics (+ Fold, model_name)
    """
    if "GroundTruth" not in df_pred.columns or "Fold" not in df_pred.columns:
        raise ValueError("df_pred must contain 'GroundTruth' and 'Fold' columns")

    fold = str(df_pred["Fold"].iloc[0])

    meta_cols = {"GroundTruth", "Fold", "compound_chembl_id", "ID", "Dataset"}
    pred_cols = [c for c in df_pred.columns if c not in meta_cols]

    # Regression: use all numeric prediction columns
    model_cols = [c for c in pred_cols if pd.api.types.is_numeric_dtype(df_pred[c])]

    rows: List[Dict[str, Any]] = []
    y_true = df_pred["GroundTruth"].values.astype(float)

    for col in model_cols:
        y_pred = df_pred[col].values.astype(float)

        # NaN/Inf guard
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t = y_true[mask]
        y_p = y_pred[mask]

        if y_t.size == 0:
            met = {}
        else:
            met = get_metrics(y_t, y_p)

        row: Dict[str, Any] = {"Fold": fold, "model_name": col}
        row.update(met)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_fold_metrics_tables(df_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_fold_metrics (pd.DataFrame): concatenated fold-level metrics tables
            Required columns: ['Fold', 'model_name', <metric columns...>]

    Returns:
        pd.DataFrame: aggregated metrics per model (mean/std over folds)
    """
    if "model_name" not in df_fold_metrics.columns:
        raise ValueError("df_fold_metrics must contain 'model_name'")

    keys = ["mse", "rmse", "r2", "pcc", "ci"]

    results: List[Dict[str, Any]] = []
    for model_name, g in df_fold_metrics.groupby("model_name"):
        row: Dict[str, Any] = {"model_name": model_name}

        for k in keys:
            if k not in g.columns:
                vals = np.asarray([], dtype=float)
            else:
                vals = g[k].astype(float).values
                vals = vals[np.isfinite(vals)]

            row[k] = float(np.mean(vals)) if vals.size > 0 else 0.0
            row[f"{k}_std"] = float(np.std(vals)) if vals.size > 0 else 0.0

        results.append(row)

    return pd.DataFrame(results)



def print_elapsed_time(start_time: float, end_time: float) -> None:
    """
    Args:
        start_time (float)
        end_time (float)

    Returns:
        None
    """
    elapsed = end_time - start_time
    total_seconds = int(elapsed)

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

    print(f"[INFO] End time: {end_time_str}")

    if days > 0:
        print(f"[INFO] Total time: {days} days {hours} hours {minutes} minutes {seconds} seconds")
    elif hours > 0:
        print(f"[INFO] Total time: {hours} hours {minutes} minutes {seconds} seconds")
    elif minutes > 0:
        print(f"[INFO] Total time: {minutes} minutes {seconds} seconds")
    else:
        print(f"[INFO] Total time: {seconds} seconds")


class MyDataset(TorchDataset):
    """
    Torch Dataset wrapper with dtype safety.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        do_normalize: bool = True,
    ):
        """
        Args:
            data (np.ndarray): Feature array (N, D)
            labels (np.ndarray): Label array (N, 1)
            do_normalize (bool)
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


class NumpyDataset(TorchDataset):
    """
    Simple torch Dataset with numpy inputs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): Feature array, shape (N, D)
            y (np.ndarray): Label array, shape (N, 1) or (N,)
        """
        self.X = X.astype(np.float32)
        y_ = y.reshape(-1, 1) if y.ndim == 1 else y
        self.y = y_.astype(np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LabelNormalizer:
    """
    Label normalizer for downstream regressors.

    Supported modes:
        - zscore
        - robust
        - minmax
        - none
    """

    def __init__(self, mode: str = "zscore"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.median = None
        self.iqr = None
        self.y_min = None
        self.y_max = None

    def fit(self, y: np.ndarray) -> None:
        y = y.astype(float)

        if self.mode == "zscore":
            self.mean = float(np.mean(y))
            self.std = float(np.std(y) + 1e-8)

        elif self.mode == "robust":
            self.median = float(np.median(y))
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            self.iqr = float(q3 - q1 + 1e-8)

        elif self.mode == "minmax":
            self.y_min = float(np.min(y))
            self.y_max = float(np.max(y))

        elif self.mode == "none":
            pass

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return (y - self.mean) / self.std

        elif self.mode == "robust":
            return (y - self.median) / self.iqr

        elif self.mode == "minmax":
            return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

        elif self.mode == "none":
            return y

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mode == "zscore":
            return z * self.std + self.mean

        elif self.mode == "robust":
            return z * self.iqr + self.median

        elif self.mode == "minmax":
            return z * (self.y_max - self.y_min) + self.y_min

        elif self.mode == "none":
            return z

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = LabelNormalizer(d["mode"])
        obj.__dict__.update(d)
        return obj