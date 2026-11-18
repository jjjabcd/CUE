"""Stage 3 — 공유 인코더(stage 1) + 불확실성(stage 2) 로 융합 임베딩을 만들고 head MLP 를 학습한다.

융합에 쓰는 모달리티 임베딩은 **항상 seed 0** 에서 나온다 (cue/core/model.py predict 와 동일).
따라서 CUE_LTAU / CUE_MC_Dropout / CUE_Deep_Ensemble 은 임베딩이 완전히 같고 가중치 w 만
다르며, UNIFORM / CONCAT 은 불확실성을 전혀 쓰지 않으므로 UQ 방법과 무관하게 셀이 하나씩이다.

    POLICY                  w_m                                   u 출처
    CUE_LTAU                softmax(-u/T)                         uncertainty/LTAU
    CUE_MC_Dropout          softmax(-u/T)                         uncertainty/MC_Dropout
    CUE_Deep_Ensemble       softmax(-u/T)                         uncertainty/Deep_Ensemble
    UNIFORM                 1/M                                   -
    CONCAT                  가중치 없이 이어붙임 (head 입력 M*D)   -

출력 (MMGW/mmgw/results 레이아웃)
---------------------------------
    results_shared/<DATASET>/<TARGET>/fusion/<MODALITY>/<POLICY>/
        metrics.csv · metrics_per_fold.csv · predictions_all.csv · config.json
        fold<k>/{fusion.csv, predictions.csv, embeddings.npy, regressors/}
    results_shared/fusion_all_metrics.csv        모든 셀을 모은 long 표

Run
---
    ~/miniconda3/envs/cue/bin/python scripts/fit_fusion.py --datasets DIS --targets PI \
        --combos FP+GNN --policies UNIFORM
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

from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs  # noqa: E402
from cue.core.downstream_runner import load_embedding_lookup  # noqa: E402
from cue.core.fusion import UncertaintyNormalizer, compute_fusion_weights  # noqa: E402
from cue.core.mlp import MLP  # noqa: E402
from cue.core.trainer import DownstreamTrainer, Trainer  # noqa: E402
from cue.data.normalization import LabelNormalizer  # noqa: E402
from cue.utils.seed import seed  # noqa: E402
from cue.utils.utils import load_fold  # noqa: E402
from train_encoders import (  # noqa: E402
    DATA_SRC, DROPOUT_RATE, HIDDEN_DIM, LABEL_MODE, TARGETS, _cache_key, ckpt_path, fold_dir,
)

# 사다리 (누적). 모달리티 추가 순서는 단일 모달리티 평균 순위를 따른다.
COMBOS = {
    "FP+GNN": ["fp", "gnn"],
    "FP+GNN+IMG": ["fp", "gnn", "img"],
    "FP+GNN+IMG+CONF": ["fp", "gnn", "img", "conf"],
    "FP+GNN+IMG+CONF+SMI": ["fp", "gnn", "img", "conf", "smi"],
}
POLICIES = ["cue_ltau", "cue_mc_dropout", "cue_deep_ensemble", "uniform", "concat"]
POLICY_UQ = {"cue_ltau": "ltau", "cue_mc_dropout": "mc_dropout",
             "cue_deep_ensemble": "deep_ensemble"}

TEMPERATURE = 1.0
UNCERTAINTY_NORM = "none"
HEAD = dict(hidden_dim_1=16, hidden_dim_2=16, hidden_dim_3=8,
            num_epochs=500, learning_rate=1e-3, batch_size=256, save_every=1)
HEAD_SEED = 1


def _embed(out_root: str, ds: str, target: str, fold: str, mod: str, fold_path: str,
           splits: Dict[str, np.ndarray], device: str) -> Dict[str, np.ndarray]:
    """멤버 m0 모달리티 MLP 의 hidden 임베딩 (N, HIDDEN_DIM)."""
    lookup = load_embedding_lookup(fold_path, _cache_key(mod))
    ck = os.path.join(fold_dir(out_root, ds, target, fold), "checkpoints")
    scaler = joblib.load(os.path.join(ck, f"scaler_{mod}.pkl"))
    x0 = np.array([lookup[s] for s in splits["train"]], dtype=np.float32)
    t = Trainer(MLP(x0.shape[1], hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE), device)
    t.load_model(ckpt_path(out_root, ds, target, fold, mod, 0))
    return {sp: t.get_embeddings(scaler.transform(
        np.array([lookup[s] for s in X], dtype=np.float32))) for sp, X in splits.items()}


def _fuse(embs: Dict[str, np.ndarray], mods: List[str], policy: str,
          u: Dict[str, np.ndarray] | None):
    """(fused, W) — cue/core/model.py predict 의 융합 분기와 같은 식."""
    n, M = embs[mods[0]].shape[0], len(mods)
    if policy == "concat":
        return np.concatenate([embs[m] for m in mods], axis=1), np.ones((n, M), np.float32)
    if policy == "uniform":
        W = np.ones((n, M), np.float32) / float(M)
    else:
        U = UncertaintyNormalizer(UNCERTAINTY_NORM)(np.stack([u[m] for m in mods], axis=1))
        W = compute_fusion_weights(u=U, policy="softmax", temperature=TEMPERATURE)
    return sum(W[:, [i]] * embs[m] for i, m in enumerate(mods)), W


def run_cell(ds: str, target: str, combo: str, policy: str, out_root: str, data_root: str,
             folds: List[str], device: str, force: bool) -> None:
    mods = COMBOS[combo]
    cell = os.path.join(out_root, ds, target, "fusion", combo, policy)
    if os.path.exists(os.path.join(cell, "metrics.csv")) and not force:
        return

    preds: List[pd.DataFrame] = []
    for fold in folds:
        fold_path = os.path.join(data_root, DATA_SRC[ds], fold)
        tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te = load_fold(fold_path, target)
        splits = {"train": X_tr, "val": X_va, "test": X_te}
        ids = {"train": tr_id, "val": va_id, "test": te_id}

        if not all(os.path.exists(ckpt_path(out_root, ds, target, fold, m, 0)) for m in mods):
            print(f"    [skip] encoders missing for {ds}/{target}/{fold}", flush=True)
            return

        embs = {m: _embed(out_root, ds, target, fold, m, fold_path, splits, device)
                for m in mods}

        u_by_split = None
        if policy in POLICY_UQ:
            uq = POLICY_UQ[policy]
            u_by_split = {}
            for sp in splits:
                f = os.path.join(out_root, ds, target, "uncertainty", uq, fold,
                                 f"{sp}_uncertainty.csv")
                if not os.path.exists(f):
                    print(f"    [skip] uncertainty missing: {f}", flush=True)
                    return
                d = pd.read_csv(f).set_index("compound_chembl_id").reindex(ids[sp])
                u_by_split[sp] = {m: d[f"u_{m}"].to_numpy(float) for m in mods}

        lab = LabelNormalizer(LABEL_MODE)
        lab.fit(y_tr)
        H, W = {}, {}
        for sp in splits:
            H[sp], W[sp] = _fuse({m: embs[m][sp] for m in mods}, mods, policy,
                                 u_by_split[sp] if u_by_split else None)

        fdir = os.path.join(cell, fold)
        os.makedirs(os.path.join(fdir, "regressors"), exist_ok=True)
        lab.save(os.path.join(fdir, "regressors", "label_normalizer.pkl"))

        # head 는 매 셀 같은 seed 로 초기화한다 (상류 RNG 소비량에 의존하지 않게).
        seed(HEAD_SEED)
        head = DownstreamMLP(input_dim=int(H["train"].shape[1]),
                             hidden_dim_1=HEAD["hidden_dim_1"], hidden_dim_2=HEAD["hidden_dim_2"],
                             hidden_dim_3=HEAD["hidden_dim_3"], output_dim=1)
        tr = DownstreamTrainer(model=head, device=torch.device(device))
        ckpt = os.path.join(fdir, "regressors", "MLP.pth")
        tr.fit(feature_name="mlp", X=H["train"],
               y=lab.transform(y_tr).reshape(-1, 1),
               batch_size=HEAD["batch_size"], num_epochs=HEAD["num_epochs"],
               learning_rate=HEAD["learning_rate"],
               validation_data={"X": H["val"], "y": lab.transform(y_va).reshape(-1, 1)},
               filepath_ckpt=ckpt, save_every=HEAD["save_every"], save_trajectory=False,
               compound_ids_tr=tr_id, compound_ids_va=va_id, smiles_tr=X_tr, smiles_va=X_va,
               task_type="regression")
        tr.load_model(ckpt)

        z = tr.inference(H["test"], batch_size=1024, task_type="regression").reshape(-1)
        df = pd.DataFrame({"Fold": fold, "compound_chembl_id": te_id,
                           target: np.asarray(y_te, float).reshape(-1),
                           "mlp": lab.inverse_transform(z).reshape(-1)})
        df.to_csv(os.path.join(fdir, "predictions.csv"), index=False)
        preds.append(df)

        fus = pd.DataFrame({"compound_chembl_id": te_id})
        for i, m in enumerate(mods):
            fus[f"w_{m}"] = W["test"][:, i]
            if u_by_split:
                fus[f"u_{m}"] = u_by_split["test"][m]
        fus.to_csv(os.path.join(fdir, "fusion.csv"), index=False)
        np.save(os.path.join(fdir, "embeddings.npy"),
                {"compound_chembl_id": te_id, "emb": H["test"], target: y_te})

    df_all = pd.concat(preds, ignore_index=True)
    df_all.to_csv(os.path.join(cell, "predictions_all.csv"), index=False)
    rows = []
    for fold in df_all.Fold.unique():
        d = df_all[df_all.Fold == fold]
        yt, yp = d[target].to_numpy(float), d["mlp"].to_numpy(float)
        rows.append(dict(Fold=fold, mse=float(np.mean((yt - yp) ** 2)),
                         rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
                         r2=float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
                         pcc=float(np.corrcoef(yt, yp)[0, 1]),
                         pred_std=float(yp.std())))
    f = pd.DataFrame(rows)
    pd.DataFrame([{"model_name": "mlp", **{f"{k}{s}": (f[k].mean() if s == "" else f[k].std(ddof=0))
                                           for k in ("mse", "rmse", "r2", "pcc")
                                           for s in ("", "_std")}}]).to_csv(
        os.path.join(cell, "metrics.csv"), index=False)
    f.to_csv(os.path.join(cell, "metrics_per_fold.csv"), index=False)
    with open(os.path.join(cell, "config.json"), "w") as fh:
        json.dump({"dataset": ds, "target": target, "combo": combo, "modalities": mods,
                   "policy": policy, "uq": POLICY_UQ.get(policy), "temperature": TEMPERATURE,
                   "uncertainty_norm": UNCERTAINTY_NORM, "head": HEAD,
                   "head_seed": HEAD_SEED, "label_mode": LABEL_MODE,
                   "encoder_seed_for_embedding": 0}, fh, indent=2)
    _append_all_metrics(out_root, ds, target, combo, policy, f)
    n_collapsed = int((f.pred_std < 1e-6).sum())
    print(f"    rmse {f.rmse.mean():.4f}  pcc {f.pcc.mean():.4f}"
          + (f"  [COLLAPSE {n_collapsed}/{len(f)} folds]" if n_collapsed else ""), flush=True)


def _append_all_metrics(out_root: str, ds: str, target: str, combo: str, policy: str,
                        per_fold: pd.DataFrame) -> None:
    """모든 셀의 fold 별 지표를 long 표 하나에 모은다 (MMGW all_metrics.csv 와 같은 역할)."""
    path = os.path.join(out_root, "fusion_all_metrics.csv")
    rows = per_fold.assign(dataset=ds, target=target, modality=combo, policy=policy,
                           uq=POLICY_UQ.get(policy, ""))[
        ["dataset", "target", "modality", "policy", "uq", "Fold",
         "mse", "rmse", "r2", "pcc", "pred_std"]]
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = ["dataset", "target", "modality", "policy", "Fold"]
        old = old.merge(rows[key].assign(_drop=1), on=key, how="left")
        old = old[old._drop.isna()].drop(columns="_drop")
        rows = pd.concat([old, rows], ignore_index=True)
    rows.sort_values(["dataset", "target", "modality", "policy", "Fold"]).to_csv(
        path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=os.path.join(REPO, "results_shared"))
    p.add_argument("--data_root", default=os.path.join(
        os.path.dirname(REPO), "chembl", "data", "selectivity_df"))
    p.add_argument("--datasets", nargs="+", default=list(DATA_SRC))
    p.add_argument("--targets", nargs="+", default=TARGETS)
    p.add_argument("--combos", nargs="+", default=list(COMBOS))
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--folds", nargs="+", default=[f"fold{k}" for k in range(1, 6)])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    cells = [(ds, t, c, pol) for ds in a.datasets for t in a.targets
             for c in a.combos for pol in a.policies]
    t0 = time.time()
    for i, (ds, t, c, pol) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {ds}/{t}/{c}/{pol}  (+{time.time() - t0:.0f}s)", flush=True)
        run_cell(ds, t, c, pol, a.out_root, a.data_root, a.folds, a.device, a.force)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
