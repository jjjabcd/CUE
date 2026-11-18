from __future__ import annotations

import re
import time
import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import torch
import tqdm

from cue.core.model import CUE, CUEConfigs
from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs
from cue.core.trainer import DownstreamTrainer
from cue.data.normalization import LabelNormalizer
from cue.utils.seed import seed
from cue.utils.metric import get_metrics

from cue.utils.utils import (
    ensure_dirs, load_fold, aggregate_metrics
)
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*Asking to truncate to max_length.*")


def parse_args() -> argparse.Namespace:
    """
    Args:
        None

    Returns:
        argparse.Namespace
    """
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--cue_source_dir", type=str, required=True,
        help="Existing run dir (e.g. results/<data>/<target>/<version>/CUE) whose "
             "per-fold checkpoints/ AND ltau/{fp,img}/{train,val,test}_uncertainty.csv "
             "are reused as-is. Only the fusion policy/temperature and the downstream "
             "head are (re)computed. Must have been produced with the same "
             "data_dir/target_col/hidden_dim/task_type/ltau_bins/ltau_k.",
    )
    p.add_argument("--target_col", type=str, default="Ssel")
    p.add_argument("--label_mode", type=str, default="zscore")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    # Task type
    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])

    # Fusion
    p.add_argument("--fusion_policy", type=str, default="softmax", choices=["softmax", "random", "uniform", "concat", "sum", "custom"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--uncertainty_norm", type=str, default="none", choices=["none", "minmax", "zscore"])
    p.add_argument("--custom_weight", type=float, nargs=2, default=None, metavar=("FP_WEIGHT", "IMG_WEIGHT"), help="Fixed [fp, img] weight, required when fusion_policy='custom'")

    # CUE pretrain configs (must match cue_source_dir; not retrained here, kept for record/model-shape)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_every", type=int, default=1)

    # CUE Modalities
    p.add_argument("--active_modalities", type=str, nargs="+", default=["fp", "img"], help="List of modalities to use")

    # Downstream MLP regressor/classifier head
    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)
    p.add_argument("--reg_num_epochs", type=int, default=500)
    p.add_argument("--reg_learning_rate", type=float, default=1e-3)
    p.add_argument("--reg_batch_size", type=int, default=256)
    p.add_argument("--reg_save_every", type=int, default=1)

    return p.parse_args()


def _load_uncertainty(csv_path: str, ids: np.ndarray) -> np.ndarray:
    """
    Args:
        csv_path (str): path to a {train,val,test}_uncertainty.csv from a source LTAU run
        ids (np.ndarray): compound_chembl_id order to align to

    Returns:
        np.ndarray: 'uncertainty' column values, reindexed to match `ids` order
    """
    df = pd.read_csv(csv_path).set_index("compound_chembl_id")
    return df.loc[ids, "uncertainty"].values.astype(float)


def main() -> None:
    start_time = time.time()
    print(f"[INFO] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("=" * 80)

    args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)
    print(f"[INFO] Config saved to {config_path}")

    print("=" * 80)
    print("[START] Fusion-only training (CUE checkpoint + LTAU uncertainty reused)")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  cue_source_dir: {args.cue_source_dir}")
    print(f"  device         : {device}")
    print(f"  seed           : {args.seed}")
    print(f"  task_type      : {args.task_type}")
    print(f"  fusion_policy  : {args.fusion_policy}")
    print(f"  temperature    : {args.temperature}")
    print("=" * 80)

    all_predictions: List[pd.DataFrame] = []

    # --------------------------------------------------
    # Fold discovery (driven by the source run's folds)
    # --------------------------------------------------
    fold_dirs = sorted(
        d for d in os.listdir(args.cue_source_dir)
        if re.fullmatch(r"fold\d+", d) and os.path.isdir(os.path.join(args.cue_source_dir, d))
    )
    print(f"[INFO] Found folds: {fold_dirs}")

    for fold in tqdm.tqdm(fold_dirs, desc="Fusion folds", leave=True, dynamic_ncols=True):
        seed(args.seed)
        print("=" * 80)
        print(f"[{fold}] START")
        print("=" * 80)

        # --------------------------------------------------
        # Fold setup
        # --------------------------------------------------
        fold_path = os.path.join(args.data_dir, fold)
        fold_out = os.path.join(args.output_dir, fold)
        ensure_dirs(fold_out, args.task_type)

        source_fold_dir = os.path.join(args.cue_source_dir, fold)

        print(f"[{fold}] Loading data")
        tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te = load_fold(fold_path, args.target_col)

        print("[Data]")
        print(f"  Train: X={X_tr.shape}, y={y_tr.shape}")
        print(f"  Val  : X={X_va.shape}, y={y_va.shape}")
        print(f"  Test : X={X_te.shape}, y={y_te.shape}")

        # --------------------------------------------------
        # Label normalization (single source of truth)
        # --------------------------------------------------
        print("[LabelNormalizer]")

        if args.task_type == "classification":
            lab = LabelNormalizer("none")
        else:
            lab = LabelNormalizer(args.label_mode)

        lab.fit(y_tr)

        y_tr_n = lab.transform(y_tr)
        y_va_n = lab.transform(y_va)
        y_te_n = lab.transform(y_te)

        # --------------------------------------------------
        # CUE: load pretrained modality MLPs from source run
        # (skips CUE.fit()/save() entirely)
        # --------------------------------------------------
        print(f"[CUE] Loading pretrained modality MLPs from {source_fold_dir}")

        cue_cfg = CUEConfigs(
            fold_dir=source_fold_dir,
            device=device,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            save_every=args.save_every,
            task_type=args.task_type,
        )

        cue_model = CUE(cue_cfg)
        cue_model.load()
        print("[CUE] Loaded")

        # --------------------------------------------------
        # LTAU uncertainty: reuse from source run (no LTAU fit/estimate here)
        # --------------------------------------------------
        print(f"[LTAU] Reusing precomputed uncertainty from {source_fold_dir}/ltau")

        u_tr = {
            "fp": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "fp", "train_uncertainty.csv"), tr_id),
            "img": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "img", "train_uncertainty.csv"), tr_id),
        }
        u_va = {
            "fp": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "fp", "val_uncertainty.csv"), va_id),
            "img": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "img", "val_uncertainty.csv"), va_id),
        }
        u_te = {
            "fp": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "fp", "test_uncertainty.csv"), te_id),
            "img": _load_uncertainty(os.path.join(source_fold_dir, "ltau", "img", "test_uncertainty.csv"), te_id),
        }

        print("[LTAU] Train uncertainty stats")
        print(f"  fp  : mean={u_tr['fp'].mean():.4f}, std={u_tr['fp'].std():.4f}")
        print(f"  img : mean={u_tr['img'].mean():.4f}, std={u_tr['img'].std():.4f}")

        # CUE: Mask modalities not in args.active_modalities
        def _frozen_weights(n_samples: int) -> Any:
            if len(args.active_modalities) != 1:
                return None
            active = args.active_modalities[0]
            w_fixed = np.zeros((n_samples, 2), dtype=np.float32)
            if active == "fp":
                w_fixed[:, 0] = 1.0
            elif active == "img":
                w_fixed[:, 1] = 1.0
            return w_fixed

        frozen_weights_tr = _frozen_weights(len(X_tr))
        frozen_weights_va = _frozen_weights(len(X_va))
        frozen_weights_te = _frozen_weights(len(X_te))

        # --------------------------------------------------
        # TRAIN+VAL embedding + fusion
        # --------------------------------------------------
        print("[Embedding] Generating TRAIN+VAL embeddings")

        H_tr, W_tr, loss_tr = cue_model.predict(
            X_tr,
            y=y_tr_n,
            uncertainties=u_tr,
            fusion_policy=args.fusion_policy,
            temperature=args.temperature,
            uncertainty_norm=args.uncertainty_norm,
            random_state=args.seed,
            return_weights=True,
            return_losses=True,
            task_type=args.task_type,
            weights=frozen_weights_tr,
            custom_weight=args.custom_weight,
        )

        df_fusion_tr = pd.DataFrame({
            "compound_chembl_id": tr_id,
            "u_fp": u_tr["fp"],
            "u_img": u_tr["img"],
            "w_fp": W_tr[:, 0],
            "w_img": W_tr[:, 1],
            "loss_fp": loss_tr["loss_fp"],
            "loss_img": loss_tr["loss_img"],
        })
        df_fusion_tr.to_csv(os.path.join(fold_out, "fusion", "train_fusion.csv"), index=False)

        np.save(
            os.path.join(fold_out, "embeddings", "train_embeddings.npy"),
            {"compound_chembl_id": tr_id, "emb": H_tr, args.target_col: y_tr},
        )

        H_va, W_va, loss_va = cue_model.predict(
            X_va,
            y=y_va_n,
            uncertainties=u_va,
            fusion_policy=args.fusion_policy,
            temperature=args.temperature,
            uncertainty_norm=args.uncertainty_norm,
            random_state=args.seed,
            return_weights=True,
            return_losses=True,
            task_type=args.task_type,
            weights=frozen_weights_va,
            custom_weight=args.custom_weight,
        )

        df_fusion_va = pd.DataFrame({
            "compound_chembl_id": va_id,
            "u_fp": u_va["fp"],
            "u_img": u_va["img"],
            "w_fp": W_va[:, 0],
            "w_img": W_va[:, 1],
            "loss_fp": loss_va["loss_fp"],
            "loss_img": loss_va["loss_img"],
        })

        df_fusion_va.to_csv(os.path.join(fold_out, "fusion", "val_fusion.csv"), index=False)

        np.save(
            os.path.join(fold_out, "embeddings", "val_embeddings.npy"),
            {"compound_chembl_id": va_id, "emb": H_va, args.target_col: y_va},
        )

        # --------------------------------------------------
        # TEST embedding + fusion
        # --------------------------------------------------
        print("[Embedding] Generating TEST embeddings")

        H_te, W_te, loss_te = cue_model.predict(
            X_te,
            y=y_te_n,
            uncertainties=u_te,
            fusion_policy=args.fusion_policy,
            temperature=args.temperature,
            uncertainty_norm=args.uncertainty_norm,
            random_state=args.seed,
            return_weights=True,
            return_losses=True,
            task_type=args.task_type,
            weights=frozen_weights_te,
            custom_weight=args.custom_weight,
        )

        df_fusion_te = pd.DataFrame({
            "compound_chembl_id": te_id,
            "u_fp": u_te["fp"],
            "u_img": u_te["img"],
            "w_fp": W_te[:, 0],
            "w_img": W_te[:, 1],
            "loss_fp": loss_te["loss_fp"],
            "loss_img": loss_te["loss_img"],
        })
        df_fusion_te.to_csv(os.path.join(fold_out, "fusion", "test_fusion.csv"), index=False)

        np.save(
            os.path.join(fold_out, "embeddings", "test_embeddings.npy"),
            {"compound_chembl_id": te_id, "emb": H_te, args.target_col: y_te},
        )

        # --------------------------------------------------
        # Downstream head (MLP)
        # --------------------------------------------------
        if args.task_type == "regression":
            print("[Regressor] Training downstream MLP regressor")
            reg_dir = os.path.join(fold_out, "regressors")
            os.makedirs(reg_dir, exist_ok=True)
            lab.save(os.path.join(reg_dir, "label_normalizer.pkl"))
            mlp_ckpt_path = os.path.join(reg_dir, "MLP.pth")
        else:
            print("[Classifier] Training downstream MLP classifier")
            clf_dir = os.path.join(fold_out, "classifiers")
            os.makedirs(clf_dir, exist_ok=True)
            lab.save(os.path.join(clf_dir, "label_normalizer.pkl"))
            mlp_ckpt_path = os.path.join(clf_dir, "MLP.pth")

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
            feature_name="mlp",
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
            trajectory_dir=os.path.join(reg_dir if args.task_type == "regression" else clf_dir, "trajectories"),
            compound_ids_tr=tr_id,
            compound_ids_va=va_id,
            smiles_tr=X_tr,
            smiles_va=X_va,
            task_type=args.task_type,
            traj_error_mode="bce" if args.task_type == "classification" else "abs",
        )

        mlp_trainer.plot(
            history=history,
            save_dir=reg_dir if args.task_type == "regression" else clf_dir,
            task_type=args.task_type,
            best_key=("val_bce" if args.task_type == "classification" else "val_mse"),
        )

        mlp_trainer.load_model(mlp_ckpt_path)

        # --------------------------------------------------
        # Test prediction (Internal test)
        # --------------------------------------------------
        print("=" * 80 + "\n")
        print("[Prediction] Predicting test set")

        z_pred_te_mlp = mlp_trainer.inference(
            H_te,
            batch_size=1024,
            task_type=args.task_type,
            return_logits=True,
        )

        fold_records: List[Dict[str, Any]] = []
        for i, (cid, gt, h) in enumerate(zip(te_id, y_te, H_te)):
            row: Dict[str, Any] = {
                "compound_chembl_id": cid,
                "GroundTruth": float(gt),
                "Fold": fold,
            }

            if args.task_type == "regression":
                row["mlp"] = float(lab.inverse_transform(float(z_pred_te_mlp[i])))
            else:
                logit = float(z_pred_te_mlp[i])
                prob = float(1.0 / (1.0 + np.exp(-logit)))
                row["mlp"] = prob

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

    if args.task_type == "regression":
        df_metrics = aggregate_metrics(df_all, task_type="regression")
        df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
        df_all.to_csv(os.path.join(args.output_dir, "predictions_all.csv"), index=False)

    else:
        meta_cols = {"GroundTruth", "Fold", "compound_chembl_id", "ID", "Dataset"}
        prob_cols = [c for c in df_all.columns if (c not in meta_cols) and (c.endswith("_prob") or c == "mlp_prob" or c == "mlp")]

        if len(prob_cols) == 0:
            prob_cols = [c for c in df_all.columns if c not in meta_cols]

        fold_metrics_rows: List[Dict[str, Any]] = []

        folds = sorted(df_all["Fold"].unique())
        for fold in folds:
            df_f = df_all[df_all["Fold"] == fold].reset_index(drop=True)
            y_true = df_f["GroundTruth"].values.astype(float)

            fold_pred_cols = [c for c in prob_cols if c in df_f.columns]

            for col in fold_pred_cols:
                try:
                    y_score = df_f[col].values.astype(float)
                except Exception:
                    continue

                met = get_metrics(y_true, y_score, task_type="classification")

                row: Dict[str, Any] = {
                    "Fold": fold,
                    "model_name": col,
                }
                row.update(met)
                fold_metrics_rows.append(row)

        if len(fold_metrics_rows) == 0:
            print("[WARN] No classification prediction columns found to aggregate metrics.")
            df_metrics = pd.DataFrame(
                columns=["model_name", "auroc", "auroc_std", "auprc", "auprc_std", "f1", "f1_std", "accuracy", "accuracy_std"]
            )
            df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
            df_all.to_csv(os.path.join(args.output_dir, "predictions_all.csv"), index=False)

        else:
            df_fold_metrics_all = pd.DataFrame(fold_metrics_rows)

            metric_keys = ["auroc", "auprc", "f1", "accuracy"]
            agg_rows: List[Dict[str, Any]] = []

            for model_name in sorted(df_fold_metrics_all["model_name"].unique()):
                sub = df_fold_metrics_all[df_fold_metrics_all["model_name"] == model_name]

                row: Dict[str, Any] = {"model_name": model_name}
                for k in metric_keys:
                    vals = sub[k].astype(float).values if k in sub.columns else np.array([], dtype=float)
                    row[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                    row[f"{k}_std"] = float(np.std(vals)) if len(vals) > 0 else 0.0
                agg_rows.append(row)

            df_metrics = pd.DataFrame(agg_rows)
            df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
            df_all.to_csv(os.path.join(args.output_dir, "predictions_all.csv"), index=False)

    print("[DONE] All folds completed")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
