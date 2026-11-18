"""Re-fuse an already-trained modality ladder cell under a different fusion policy.

왜 이 스크립트가 필요한가
-------------------------
scripts/run_fusion_ablation.sh 는 융합 정책만 바꾸려고 파이프라인 전체를 다시 돌린다.
그런데 모달리티 MLP 사전학습(1000 epoch x 모달리티 수)이 런타임의 대부분이고, 융합 정책은
그보다 윗 단계라 인코더에 아무 영향이 없다 — v1(softmax) 셀과 uniform 셀의 history_*.csv 가
fold 단위로 비트 단위까지 같다는 것을 확인했다. 즉 재학습은 전부 낭비다.

이 스크립트는 기존 v1 셀에 저장된 것을 그대로 재사용한다.
  checkpoints/mlp_<mod>.pth, scaler_<mod>.pkl  -> CUE.load() 로 복원
  fusion/{train,val,test}_fusion.csv 의 u_<mod> -> 저장된 불확실성 재사용(LTAU 재추정 생략)
그 위에서 융합만 새 정책으로 다시 하고, 다운스트림 헤드(500 epoch, 작은 MLP)만 새로 학습한다.

uniform / concat 에서 불확실성은 융합 가중치에 쓰이지 않는다(uniform 은 1/M 고정, concat 은
u >= 1e8 마스킹에만 사용). 저장된 u 는 모두 1e8 미만이므로 마스킹도 일어나지 않는다. 따라서
저장된 u 를 그대로 넘기는 것이 원래 파이프라인과 동일하다.

사용법
------
    ~/miniconda3/envs/cue/bin/python scripts/refit_fusion.py --policy concat
    ~/miniconda3/envs/cue/bin/python scripts/refit_fusion.py \
        --policy uniform --out_version UNIFORM_REFIT

출력: results/<UQ>/<ds>/<target>/<modality>/<out_version>/
      fold*/{predictions.csv, fusion/*.csv, embeddings/*.npy, regressors/*}
      predictions_all.csv, metrics.csv, config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cue.core.model import CUE, CUEConfigs  # noqa: E402
from cue.core.downstream_mlp import DownstreamMLP  # noqa: E402
from cue.core.trainer import DownstreamTrainer  # noqa: E402
from cue.data.normalization import LabelNormalizer  # noqa: E402
from cue.utils.seed import seed as set_seed  # noqa: E402
from cue.utils.utils import LEGACY_COLUMNS, aggregate_metrics, load_fold  # noqa: E402

SPLITS = ("train", "val", "test")


def read_saved_uncertainty(fold_src: str, split: str, mods: List[str],
                           ids: np.ndarray) -> Dict[str, np.ndarray]:
    """v1 셀이 저장해 둔 u_<mod> 를 compound id 순서에 맞춰 돌려준다."""
    path = os.path.join(fold_src, "fusion", f"{split}_fusion.csv")
    df = pd.read_csv(path)
    if len(df) != len(ids) or not (df["compound_chembl_id"].to_numpy() == ids).all():
        # 순서가 다르면 id 로 정렬해 맞춘다(순서 불일치는 조용한 오염이 되므로 반드시 처리).
        df = df.set_index("compound_chembl_id").loc[ids].reset_index()
    return {m: df[f"u_{m}"].to_numpy(dtype=float) for m in mods}


def refit_fold(cell_src: str, cell_dst: str, fold: str, cfg: dict, args) -> pd.DataFrame:
    fold_src = os.path.join(cell_src, fold)
    fold_dst = os.path.join(cell_dst, fold)
    os.makedirs(os.path.join(fold_dst, "fusion"), exist_ok=True)
    os.makedirs(os.path.join(fold_dst, "embeddings"), exist_ok=True)
    reg_dir = os.path.join(fold_dst, "regressors")
    os.makedirs(reg_dir, exist_ok=True)

    device = torch.device(args.device)
    mods = list(cfg["active_modalities"])
    # v1 셀 중 이름 통일 이전에 돌린 것들은 config 에 레거시 타깃명을 들고 있다
    # (WS(alpha=2), RS(k=2)). load_fold 는 컬럼을 통일된 이름으로 바꿔 읽으므로 여기서도 맞춘다.
    target_col = LEGACY_COLUMNS.get(cfg["target_col"], cfg["target_col"])
    data_fold = os.path.normpath(os.path.join(REPO, cfg["data_dir"], fold))

    tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te = load_fold(data_fold, target_col)

    # 학습된 모달리티 MLP 복원. fold_dir 은 체크포인트가 있는 v1 쪽을 가리켜야 한다.
    cue_cfg = CUEConfigs(
        fold_dir=fold_src,
        device=device,
        modalities=mods,
        data_fold_dir=data_fold,
        gnn_backend=cfg["gnn_backend"],
        hidden_dim=int(cfg["hidden_dim"]),
        batch_size=int(cfg["batch_size"]),
        task_type=cfg["task_type"],
        uncertainty_method=cfg["uncertainty_method"],
        dropout_rate=float(cfg["dropout_rate"]),
        ensemble_size=int(cfg["ensemble_size"]),
        ensemble_seeds=list(cfg["ensemble_seeds"]),
        evidential_lambda=float(cfg["evidential_lambda"]),
        evidential_uncertainty=cfg["evidential_uncertainty"],
    )
    cue_model = CUE(cue_cfg)
    cue_model.load()

    lab = LabelNormalizer(cfg.get("label_mode", "none"))
    lab.fit(y_tr)
    y_tr_n, y_va_n, y_te_n = lab.transform(y_tr), lab.transform(y_va), lab.transform(y_te)

    H, W, losses = {}, {}, {}
    for split, X, y_n, ids in (("train", X_tr, y_tr_n, tr_id),
                               ("val", X_va, y_va_n, va_id),
                               ("test", X_te, y_te_n, te_id)):
        u = read_saved_uncertainty(fold_src, split, mods, ids)
        H[split], W[split], losses[split] = cue_model.predict(
            X, y=y_n, uncertainties=u,
            fusion_policy=args.policy,
            temperature=float(cfg["temperature"]),
            uncertainty_norm=cfg["uncertainty_norm"],
            random_state=int(cfg["seed"]),
            return_weights=True, return_losses=True,
            task_type=cfg["task_type"],
        )
        df_f = pd.DataFrame({"compound_chembl_id": ids})
        for i, m in enumerate(mods):
            df_f[f"u_{m}"] = u[m]
            df_f[f"w_{m}"] = W[split][:, i]
            df_f[f"loss_{m}"] = losses[split][f"loss_{m}"]
        df_f.to_csv(os.path.join(fold_dst, "fusion", f"{split}_fusion.csv"), index=False)

    y_raw = {"train": y_tr, "val": y_va, "test": y_te}
    ids_all = {"train": tr_id, "val": va_id, "test": te_id}
    for split in SPLITS:
        np.save(os.path.join(fold_dst, "embeddings", f"{split}_embeddings.npy"),
                {"compound_chembl_id": ids_all[split], "emb": H[split],
                 target_col: y_raw[split]})

    # 다운스트림 헤드만 새로 학습. concat 은 입력 차원이 M*D 로 커지므로 H 에서 읽는다.
    set_seed(int(args.head_seed))
    model = DownstreamMLP(
        input_dim=int(H["train"].shape[1]),
        hidden_dim_1=int(cfg["reg_hidden_dim_1"]),
        hidden_dim_2=int(cfg["reg_hidden_dim_2"]),
        hidden_dim_3=int(cfg["reg_hidden_dim_3"]),
        output_dim=1,
    )
    trainer = DownstreamTrainer(model=model, device=device)
    ckpt = os.path.join(reg_dir, "MLP.pth")
    history = trainer.fit(
        feature_name="mlp", X=H["train"],
        y=y_tr_n.reshape(-1, 1) if y_tr_n.ndim == 1 else y_tr_n,
        batch_size=int(cfg["reg_batch_size"]), num_epochs=int(cfg["reg_num_epochs"]),
        learning_rate=float(cfg["reg_learning_rate"]),
        validation_data={"X": H["val"],
                         "y": y_va_n.reshape(-1, 1) if y_va_n.ndim == 1 else y_va_n},
        filepath_ckpt=ckpt, save_every=int(cfg["reg_save_every"]),
        save_trajectory=False, task_type="regression", traj_error_mode="abs",
    )
    trainer.plot(history=history, save_dir=reg_dir, task_type="regression",
                 best_key="val_mse")
    trainer.load_model(ckpt)
    lab.save(os.path.join(reg_dir, "label_normalizer.pkl"))

    z_te = trainer.inference(H["test"], batch_size=1024, task_type="regression",
                             return_logits=True)
    df_fold = pd.DataFrame([
        {"compound_chembl_id": c, "GroundTruth": float(g), "Fold": fold,
         "mlp": float(lab.inverse_transform(float(z_te[i])))}
        for i, (c, g) in enumerate(zip(te_id, y_te))
    ])
    df_fold.to_csv(os.path.join(fold_dst, "predictions.csv"), index=False)
    return df_fold


def refit_cell(cell_src: str, args) -> dict:
    cfg = json.load(open(os.path.join(cell_src, "config.json")))
    # <...>/<modality>/<src_variant>  ->  <...>/<modality>/<out_variant>
    parts = cell_src.split(os.sep)
    cell_dst = os.sep.join(parts[:-1] + [args.out_version])
    if os.path.exists(os.path.join(cell_dst, "metrics.csv")) and not args.force:
        print(f"  [SKIP] {cell_dst}")
        return {}
    os.makedirs(cell_dst, exist_ok=True)

    folds = sorted(d for d in os.listdir(cell_src)
                   if d.startswith("fold") and os.path.isdir(os.path.join(cell_src, d)))
    t0 = time.time()
    frames = [refit_fold(cell_src, cell_dst, f, cfg, args) for f in folds]
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(os.path.join(cell_dst, "predictions_all.csv"), index=False)
    df_m = aggregate_metrics(df_all, task_type="regression")
    df_m.to_csv(os.path.join(cell_dst, "metrics.csv"), index=False)

    new_cfg = dict(cfg)
    new_cfg.update({"fusion_policy": args.policy, "output_dir": cell_dst,
                    "head_seed": int(args.head_seed),
                    "refit_from": cell_src,
                    "note": "modality MLPs reused from refit_from; only fusion + head redone"})
    json.dump(new_cfg, open(os.path.join(cell_dst, "config.json"), "w"), indent=2)

    row = df_m[df_m.iloc[:, 0].astype(str) == "mlp"]
    rmse = float(row["rmse"].iloc[0]) if len(row) and "rmse" in row else float("nan")
    print(f"  [done] {cell_dst}  rmse={rmse:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return {"cell": cell_dst, "rmse": rmse}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=os.path.join(REPO, "results"))
    p.add_argument("--uq", default="LTAU", help="재사용할 UQ 디렉터리 (기본 LTAU)")
    p.add_argument("--src_variant", default="CUE",
                   help="재사용할 변종(기본 CUE = softmax 융합)")
    p.add_argument("--policy", required=True, choices=["uniform", "concat", "sum"])
    p.add_argument("--out_version", default=None,
                   help="출력 변종 이름. 기본값은 --policy 를 대문자로 (uniform -> UNIFORM)")
    p.add_argument("--head_seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--include", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.out_version = args.out_version or args.policy.upper()

    # 통합 레이아웃: <UQ>/<ds>/<target>/<modality>/<variant>
    root = os.path.join(args.results_root, args.uq)
    cells = sorted(
        dp for dp, dn, fn in os.walk(root)
        if "config.json" in fn and any(d.startswith("fold") for d in dn)
        and os.path.basename(dp) == args.src_variant
    )
    if args.include:
        cells = [c for c in cells if args.include in c]
    if not cells:
        raise SystemExit(f"no {args.src_variant} cells under {root}")

    print(f"[INFO] {len(cells)} cell(s), {args.src_variant} -> {args.out_version} "
          f"(policy={args.policy})",
          flush=True)
    t0 = time.time()
    rows = []
    for i, c in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {c}", flush=True)
        r = refit_cell(c, args)
        if r:
            rows.append(r)
    print(f"\n[INFO] {len(rows)} cell(s) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
