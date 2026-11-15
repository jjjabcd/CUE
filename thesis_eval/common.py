"""Shared helpers for running a baseline on the thesis CUE / TDC folds.

This file is intentionally identical in frame/thesis_eval/ and molecular-property-fusion/thesis_eval/
(the two projects are separate repositories with separate environments, so it is copied, not imported).

Protocol (same for every baseline, see thesis/docs/design/v3/results_{cue,tdc}.md):
  * folds:   data/cue/processed/{DIS,INH}/fold{1..5}/{train,val,test}.csv
             data/tdc/processed/<ds>/<ds>/fold{1..5}/{train,val,test}.csv
  * labels:  thesis/src/data/normalization.py::LabelNormalizer, fitted on the TRAIN labels of the fold only.
             TDC datasets use their DATASET_MODES entry (zscore / log_zscore); CUE uses zscore.
  * metrics: computed on the ORIGINAL label scale from inverse-transformed predictions.
"""
from __future__ import annotations

import importlib.util
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

THESIS_ROOT = Path(__file__).resolve().parents[2]
CUE_ROOT = THESIS_ROOT / "data" / "cue" / "processed"
TDC_ROOT = THESIS_ROOT / "data" / "tdc" / "processed"
RESULT_VERSION = "v4"
FOLDS = (1, 2, 3, 4, 5)

CUE_ASSAYS = ("DIS", "INH")
CUE_TARGETS = ("PI", "RS2", "Ssel", "WS2")
TDC_DATASETS = (
    "Caco2_Wang",
    "Lipophilicity_AstraZeneca",
    "VDss_Lombardo",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "LD50_Zhu",
)
MPF_DEFAULT_MAX_ATOMS = 115


@dataclass(frozen=True)
class Task:
    group: str  # "cue" | "tdc"
    dataset: str  # DIS / INH / <tdc name>
    target: str  # label column (== dataset for TDC)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.target}"

    def fold_dir(self, fold: int) -> Path:
        if self.group == "cue":
            return CUE_ROOT / self.dataset / f"fold{fold}"
        return TDC_ROOT / self.dataset / self.dataset / f"fold{fold}"

    @property
    def normalizer_dataset_name(self) -> str | None:
        return self.dataset if self.group == "tdc" else None


def all_tasks(group: str | None = None) -> list[Task]:
    tasks = []
    if group in (None, "cue"):
        tasks += [Task("cue", a, t) for a in CUE_ASSAYS for t in CUE_TARGETS]
    if group in (None, "tdc"):
        tasks += [Task("tdc", d, d) for d in TDC_DATASETS]
    return tasks


def get_task(group: str, dataset: str, target: str | None = None) -> Task:
    target = target or (dataset if group == "tdc" else None)
    task = Task(group, dataset, target)
    if task not in all_tasks(group):
        raise ValueError(f"unknown task {task}")
    return task


def load_fold(task: Task, fold: int) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} frames with columns id, smiles, y (original scale)."""
    out = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(task.fold_dir(fold) / f"{split}.csv")
        df = df[["id", "smiles", task.target]].rename(columns={task.target: "y"})
        if df["y"].isna().any():
            raise ValueError(f"{task.key} fold{fold} {split}: NaN labels")
        out[split] = df.reset_index(drop=True)
    return out


def max_atoms_for(task: Task) -> int:
    """MPF graph capacity: the paper default (115), or the task-wide maximum atom count if larger.

    Scans every fold/split of the task, so every molecule that any run will see is covered.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    top = 0
    for fold in FOLDS:
        for df in load_fold(task, fold).values():
            for smi in df["smiles"]:
                top = max(top, Chem.MolFromSmiles(smi).GetNumAtoms())
    return max(MPF_DEFAULT_MAX_ATOMS, top)


def _load_label_normalizer():
    path = THESIS_ROOT / "thesis" / "src" / "data" / "normalization.py"
    spec = importlib.util.spec_from_file_location("thesis_label_normalization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LabelNormalizer


def make_normalizer(task: Task, y_train: np.ndarray):
    """LabelNormalizer fitted on the fold's training labels only."""
    norm = _load_label_normalizer()(dataset_name=task.normalizer_dataset_name, mode="zscore")
    norm.fit(np.asarray(y_train, dtype=float))
    return norm


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pcc": float(pearsonr(y_true, y_pred)[0]),
        "srcc": float(spearmanr(y_true, y_pred)[0]),
    }


def result_dir(project_root: Path, task: Task, fold: int | None = None, version: str = RESULT_VERSION) -> Path:
    base = project_root / "results" / version / task.dataset / task.target
    return base if fold is None else base / f"fold{fold}"


def git_state(path: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty_files": (run("status", "--short") or "").splitlines()}


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "host": socket.gethostname(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)
    return info


def write_fold_result(
    out_dir: Path,
    test: pd.DataFrame,
    pred_norm: np.ndarray,
    norm,
    meta: dict,
    ids_predicted: list | None = None,
) -> dict:
    """Write test_preds.csv, metrics.csv, run.json for one fold; metrics on the original label scale.

    `test` has columns id, smiles, y (original scale). If a model could not predict some test rows,
    pass the ids it did predict via `ids_predicted`; the metrics then cover that subset only and the
    counts are recorded (nothing is dropped silently).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = test if ids_predicted is None else test[test["id"].isin(set(ids_predicted))]
    frame = frame.set_index("id").loc[ids_predicted if ids_predicted is not None else test["id"]].reset_index()
    pred_norm = np.asarray(pred_norm, float)
    pred = norm.inverse_transform(pred_norm)
    preds = pd.DataFrame(
        {"id": frame["id"], "y_true": frame["y"].to_numpy(), "y_pred": pred, "y_pred_norm": pred_norm}
    )
    preds.to_csv(out_dir / "test_preds.csv", index=False)
    metrics = regression_metrics(preds["y_true"], preds["y_pred"])
    metrics.update(n_test=len(test), n_predicted=len(preds))
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    meta = {
        **meta,
        "label_normalizer": {"mode": norm.mode, "mean": norm.mean, "std": norm.std},
        "metrics": metrics,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    return metrics


def aggregate_task(project_root: Path, task: Task, version: str = RESULT_VERSION) -> Path | None:
    """results/<v>/<dataset>/<target>/metrics.csv: one row per finished fold plus mean/std."""
    rows = []
    for fold in FOLDS:
        path = result_dir(project_root, task, fold, version) / "metrics.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            rows.append({"fold": f"fold{fold}", **row})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    num = df.drop(columns="fold")
    summary = pd.DataFrame(
        [{"fold": "mean", **num.mean().to_dict()}, {"fold": "std", **num.std(ddof=1).to_dict()}]
    )
    out = result_dir(project_root, task, None, version) / "metrics.csv"
    pd.concat([df, summary], ignore_index=True).to_csv(out, index=False)
    return out
