"""Run CUE (Chemical Uncertainty-aware Embedding; FP + MolScribe-image fusion with LTAU) on the thesis TDC folds.

The upstream code (train.py, cue/) is used unchanged with the defaults of scripts/run.sh (hidden 32, 1000 epochs,
lr 1e-4, LTAU k=5 / 5000 bins, regressor 16-16-8, 500 epochs, seed 2025). Wrapper-level choices (recorded in run.json):
  * the thesis fold split (data/tdc/processed/<ds>/<ds>/fold{k}/{train,val,test}.csv) is written in CUE's input format
    (selectivity_{train,val,test}.csv with compound_chembl_id, SMILES, <label>);
  * labels are normalised with the thesis LabelNormalizer (zscore / log_zscore, fitted on the fold's train labels) and
    CUE is run with --label_mode none on these labels, because CUE's own LabelNormalizer has no log_zscore mode;
  * predictions (column `mlp` of fold{k}/predictions.csv) are inverse-transformed and the metrics are computed on the
    original label scale.

Example:
  python thesis_eval/run_cue_tdc.py --dataset Caco2_Wang --folds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

# scripts/run.sh defaults, passed explicitly so that run.json records them
CUE_ARGS = dict(
    seed=2025, temperature=1.0, ltau_bins=5000, ltau_k=5,
    hidden_dim=32, num_epochs=1000, learning_rate=1e-4, batch_size=256, save_every=1,
    reg_hidden_dim_1=16, reg_hidden_dim_2=16, reg_hidden_dim_3=8, reg_num_epochs=500,
    reg_learning_rate=1e-3, reg_batch_size=256, reg_save_every=1,
)
SMOKE_ARGS = dict(num_epochs=2, reg_num_epochs=2)


def version_of(args) -> str:
    return "v4_smoke" if args.smoke else C.RESULT_VERSION


def write_cue_input(root: Path, fold: int, task: C.Task, data: dict, norm) -> Path:
    fold_dir = root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("val", "val"), ("test", "test")):
        df = data[split]
        pd.DataFrame(
            {"compound_chembl_id": df["id"], "SMILES": df["smiles"], task.target: norm.transform(df["y"].to_numpy())}
        ).to_csv(fold_dir / f"selectivity_{name}.csv", index=False)
    return root


def run_fold(task: C.Task, fold: int, args) -> dict:
    out_dir = C.result_dir(PROJECT, task, fold, version_of(args))
    if (out_dir / "metrics.csv").exists() and not args.overwrite:
        print(f"[skip] {task.key} fold{fold}: results exist")
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    data = C.load_fold(task, fold)
    norm = C.make_normalizer(task, data["train"]["y"].to_numpy())
    input_root = out_dir / "_cue_input"
    write_cue_input(input_root, fold, task, data, norm)

    cue_args = {**CUE_ARGS, **(SMOKE_ARGS if args.smoke else {})}
    cue_out = out_dir / "cue_output"
    cmd = [sys.executable, "train.py", "--data_dir", str(input_root), "--output_dir", str(cue_out),
           "--target_col", task.target, "--label_mode", "none", "--device", args.device]
    for k, v in cue_args.items():
        cmd += [f"--{k}", str(v)]
    started = datetime.now().astimezone()
    t0 = time.time()
    with open(out_dir / "cue.log", "a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        subprocess.run(cmd, cwd=PROJECT, stdout=fh, stderr=subprocess.STDOUT, check=True)
    elapsed = time.time() - t0

    pred = pd.read_csv(cue_out / f"fold{fold}" / "predictions.csv")
    pred = pred.set_index("compound_chembl_id").loc[data["test"]["id"]]
    assert len(pred) == len(data["test"])
    assert np.allclose(pred["GroundTruth"].to_numpy(), norm.transform(data["test"]["y"].to_numpy()), atol=1e-5)
    meta = {
        "model": "CUE", "task": task.key, "fold": fold, "smoke": bool(args.smoke),
        "cue_args": {**cue_args, "label_mode": "none (labels pre-normalised by the thesis LabelNormalizer)"},
        "n_train": len(data["train"]), "n_val": len(data["val"]),
        "timing_seconds": {"total": elapsed},
        "started": started.isoformat(), "ended": datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv), "cue_command": " ".join(cmd),
        "code": C.git_state(PROJECT), "env": C.environment_info(),
    }
    metrics = C.write_fold_result(out_dir, data["test"], pred["mlp"].to_numpy(), norm, meta)
    print(f"[done] {task.key} fold{fold}: {metrics}")
    if not args.keep_intermediate:
        shutil.rmtree(input_root, ignore_errors=True)
        for sub in ("trajectories", "emb"):  # large intermediates; checkpoints/ltau/fusion are kept
            shutil.rmtree(cue_out / f"fold{fold}" / sub, ignore_errors=True)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=C.TDC_DATASETS)
    p.add_argument("--folds", type=int, nargs="+", default=list(C.FOLDS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true", help="2 epochs (software check, not a result)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    args = p.parse_args()
    task = C.get_task("tdc", args.dataset)
    for fold in args.folds:
        run_fold(task, fold, args)
    print("aggregate:", C.aggregate_task(PROJECT, task, version_of(args)))


if __name__ == "__main__":
    main()
