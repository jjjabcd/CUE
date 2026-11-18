"""Stage 2 — 공유 인코더(stage 1)에서 멤버 예측과 UQ 방법 3개의 불확실성 u 를 만든다.

세 방법 모두 **같은 MLP** 를 쓴다 (stage 1 이 학습한 m0..m9). 다른 것은 u 를 내는 방식뿐이다:

    LTAU           m0 의 학습 궤적(train_trajectories.csv) + m0 의 hidden 임베딩으로 KNN
                   추정기를 적합한 뒤 train/val/test 에 적용
    MC Dropout     m0 에서 dropout 을 켠 채 T 회 forward -> 예측 분산
    Deep Ensemble  m0..m9 의 결정론적 예측 분산

u 는 **모달리티 단위**라 사다리 단계(2~5 모달)와 무관하다. 한 번만 계산해 조합이 공유한다.

출력 (MMGW/mmgw/results 레이아웃)
---------------------------------
    results_shared/<DATASET>/<TARGET>/predictions/fold<k>/member_<i>/
        {train,val,test}_predictions.csv          열: compound_chembl_id, <mod>_pred ...
    results_shared/<DATASET>/<TARGET>/uncertainty/<uq>/fold<k>/
        {train,val,test}_uncertainty.csv          열: compound_chembl_id, u_<mod> ...
        config.json

Run
---
    ~/miniconda3/envs/cue/bin/python scripts/compute_uncertainty.py --datasets DIS --targets PI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cue.core.downstream_runner import load_embedding_lookup  # noqa: E402
from cue.core.mlp import MLP  # noqa: E402
from cue.core.trainer import Trainer, predict_ensemble  # noqa: E402
from cue.ltau.pipeline import (  # noqa: E402
    estimate_uncertainty_for_split,
    fit_ltau_estimator_from_train_trajectories,
)
from cue.utils.utils import load_fold  # noqa: E402
from train_encoders import (  # noqa: E402
    DATA_SRC, DROPOUT_RATE, HIDDEN_DIM, MODALITIES, N_MEMBERS, TARGETS,
    _cache_key, ckpt_path, fold_dir,
)

UQ_METHODS = ["ltau", "mc_dropout", "deep_ensemble"]
SPLITS = ("train", "val", "test")
MC_SAMPLES = 10           # 기존 run_*.sh 와 동일
LTAU_BINS = 5000
LTAU_K = 5


def _out_dirs(out_root: str, ds: str, target: str, fold: str):
    base = os.path.join(out_root, ds, target)
    return (os.path.join(base, "predictions", fold),
            {uq: os.path.join(base, "uncertainty", uq, fold) for uq in UQ_METHODS})


def run_fold(ds: str, target: str, fold: str, mods: List[str], out_root: str,
             data_root: str, device: str, force: bool) -> None:
    """한 fold 의 모든 모달리티에 대해 멤버 예측 + u 3종을 만든다 (wide CSV)."""
    missing = [m for m in mods
               if not os.path.exists(ckpt_path(out_root, ds, target, fold, m, N_MEMBERS - 1))]
    if missing:
        print(f"    [skip] encoders incomplete: {missing}", flush=True)
        return

    pred_dir, u_dirs = _out_dirs(out_root, ds, target, fold)
    done = all(os.path.exists(os.path.join(u_dirs[uq], f"{sp}_uncertainty.csv"))
               for uq in UQ_METHODS for sp in SPLITS)
    if done and not force:
        return

    fold_path = os.path.join(data_root, DATA_SRC[ds], fold)
    tr_id, va_id, te_id, X_tr, _, X_va, _, X_te, _ = load_fold(fold_path, target)
    ids = {"train": tr_id, "val": va_id, "test": te_id}
    smiles = {"train": X_tr, "val": X_va, "test": X_te}
    ck = os.path.join(fold_dir(out_root, ds, target, fold), "checkpoints")

    # {split: {mod: (N,)}} 로 모았다가 마지막에 wide CSV 로 쓴다.
    preds: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {i: {sp: {} for sp in SPLITS}
                                                          for i in range(N_MEMBERS)}
    u: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {uq: {sp: {} for sp in SPLITS}
                                                      for uq in UQ_METHODS}

    for mod in mods:
        lookup = load_embedding_lookup(fold_path, _cache_key(mod))
        scaler = joblib.load(os.path.join(ck, f"scaler_{mod}.pkl"))
        feats = {sp: scaler.transform(np.array([lookup[s] for s in X], dtype=np.float32))
                 for sp, X in smiles.items()}
        input_dim = feats["train"].shape[1]

        members: List[Trainer] = []
        for i in range(N_MEMBERS):
            t = Trainer(MLP(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE), device)
            t.load_model(ckpt_path(out_root, ds, target, fold, mod, i))
            members.append(t)

        # 멤버별 결정론적 예측 (정규화 타깃 공간)
        for i, t in enumerate(members):
            for sp, f in feats.items():
                preds[i][sp][mod] = np.asarray(
                    t.inference(f, batch_size=1024, task_type="regression"), float).ravel()

        # LTAU — m0 의 궤적 + m0 의 hidden 임베딩
        desc = {sp: members[0].get_embeddings(f) for sp, f in feats.items()}
        est, df_tr = fit_ltau_estimator_from_train_trajectories(
            train_csv=os.path.join(fold_dir(out_root, ds, target, fold),
                                   "trajectories", f"{mod}_m0", "train_trajectories.csv"),
            train_descriptors=desc["train"], num_bins=LTAU_BINS, k=LTAU_K, save_dir=None)
        # 궤적 CSV 의 행 순서가 load_fold 순서와 같다고 가정하지 않고 id 로 맞춘다.
        u["ltau"]["train"][mod] = (df_tr.set_index("compound_chembl_id")
                                   .reindex(ids["train"])["uncertainty"].to_numpy(float))
        for sp in ("val", "test"):
            d = estimate_uncertainty_for_split(estimator=est, split_ids=ids[sp],
                                               split_descriptors=desc[sp], save_path=None)
            u["ltau"][sp][mod] = d["uncertainty"].to_numpy(float)

        # MC Dropout — m0 에서 dropout 을 켠 T 회 forward
        for sp, f in feats.items():
            _, var = members[0].predict_mc_dropout(f, num_samples=MC_SAMPLES)
            u["mc_dropout"][sp][mod] = np.asarray(var, float).ravel()

        # Deep Ensemble — m0..m9 예측 분산
        for sp, f in feats.items():
            _, var = predict_ensemble([t.model for t in members], f, torch.device(device))
            u["deep_ensemble"][sp][mod] = np.asarray(var, float).ravel()

    for i in range(N_MEMBERS):
        d = os.path.join(pred_dir, f"member_{i}")
        os.makedirs(d, exist_ok=True)
        for sp in SPLITS:
            pd.DataFrame({"compound_chembl_id": ids[sp],
                          **{f"{m}_pred": preds[i][sp][m] for m in mods}}).to_csv(
                os.path.join(d, f"{sp}_predictions.csv"), index=False)

    for uq in UQ_METHODS:
        os.makedirs(u_dirs[uq], exist_ok=True)
        for sp in SPLITS:
            pd.DataFrame({"compound_chembl_id": ids[sp],
                          **{f"u_{m}": u[uq][sp][m] for m in mods}}).to_csv(
                os.path.join(u_dirs[uq], f"{sp}_uncertainty.csv"), index=False)
        with open(os.path.join(u_dirs[uq], "config.json"), "w") as f:
            json.dump({"uq": uq, "modalities": mods, "n_members": N_MEMBERS,
                       "mc_samples": MC_SAMPLES, "ltau_bins": LTAU_BINS, "ltau_k": LTAU_K,
                       "member_for_ltau_and_mc_dropout": 0}, f, indent=2)

    print("    ok  n_tr=%d  " % len(tr_id) + "  ".join(
        f"{uq}:{np.mean([u[uq]['test'][m].mean() for m in mods]):.3g}" for uq in UQ_METHODS),
        flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=os.path.join(REPO, "results_shared"))
    p.add_argument("--data_root", default=os.path.join(
        os.path.dirname(REPO), "chembl", "data", "selectivity_df"))
    p.add_argument("--datasets", nargs="+", default=list(DATA_SRC))
    p.add_argument("--targets", nargs="+", default=TARGETS)
    p.add_argument("--modalities", nargs="+", default=MODALITIES)
    p.add_argument("--folds", nargs="+", default=[f"fold{k}" for k in range(1, 6)])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    cells = [(ds, t, f) for ds in a.datasets for t in a.targets for f in a.folds]
    t0 = time.time()
    for i, (ds, t, f) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {ds}/{t}/{f}  (+{time.time() - t0:.0f}s)", flush=True)
        run_fold(ds, t, f, a.modalities, a.out_root, a.data_root, a.device, a.force)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
