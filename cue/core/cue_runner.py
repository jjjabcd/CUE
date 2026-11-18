"""
Shared runner for the CUE/CUE uncertainty-aware fusion pipeline (per-modality MLP
pretraining -> LTAU uncertainty -> softmax-weighted fusion -> DownstreamMLP), used by
train.py and the per-combo entry points at the repo root (FP_GNN_train.py, IMG_GNN_train.py,
FP_IMG_GNN_train.py, ...). Generalized to any subset of {fp, img, smi, gnn, conf} via
args.active_modalities - adding a new combo needs only a new thin wrapper (or none: calling
train.py directly with a different --active_modalities list already works).

smi/gnn/conf require cue.utils.prepare_features to have been run for the fold first (see
scripts/run_prepare_feature.sh) - CUE reads their embeddings from that cache rather than
recomputing them live (see cue/core/model.py).
"""
from __future__ import annotations

import re
import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm

from cue.core.evidential import EVIDENTIAL_UNCERTAINTY_TERMS
from cue.core.model import CUE, CUEConfigs, SUPPORTED_UNCERTAINTY_METHODS, SUPPORTED_GNN_BACKENDS
from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs
from cue.core.trainer import DownstreamTrainer
from cue.data.normalization import LabelNormalizer
from cue.utils.seed import seed
from cue.utils.metric import get_metrics

from cue.ltau.pipeline import (
    fit_ltau_estimator_from_train_trajectories,
    estimate_uncertainty_for_split,
)

from cue.utils.utils import ensure_dirs, load_fold, aggregate_metrics, print_elapsed_time

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*Asking to truncate to max_length.*")

ALL_MODALITIES = ["fp", "img", "smi", "gnn", "conf"]


def add_common_args(p: argparse.ArgumentParser, default_modalities: List[str] = ("fp", "img")) -> None:
    """
    Args:
        p (argparse.ArgumentParser): parser to add the shared CUE/CUE CLI options to.
        default_modalities (List[str]): default for --active_modalities (overridable from CLI).
    """
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--target_col", type=str, default="Ssel")
    p.add_argument("--label_mode", type=str, default="zscore")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])

    # Fusion
    p.add_argument("--fusion_policy", type=str, default="softmax",
                    choices=["softmax", "random", "uniform", "concat", "sum", "custom"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--uncertainty_norm", type=str, default="none", choices=["none", "minmax", "zscore"])
    p.add_argument("--custom_weight", type=float, nargs="+", default=None,
                    help="Fixed per-modality weight (same length/order as --active_modalities), required when fusion_policy='custom'")

    # Uncertainty quantification method (freely combinable with --active_modalities)
    p.add_argument(
        "--uncertainty_method",
        type=str,
        default="ltau",
        choices=list(SUPPORTED_UNCERTAINTY_METHODS),
        help="Method used to compute per-modality, per-sample uncertainty u_m(x).",
    )
    # LTAU (uncertainty_method == 'ltau')
    p.add_argument("--ltau_bins", type=int, default=5000)
    p.add_argument("--ltau_k", type=int, default=5)
    # MC Dropout (uncertainty_method == 'mc_dropout')
    p.add_argument("--mc_samples", type=int, default=10, help="MC Dropout stochastic forward passes (T).")
    p.add_argument("--dropout_rate", type=float, default=0.1, help="MC Dropout probability inside modality MLPs.")
    # Deep Ensemble (uncertainty_method == 'deep_ensemble')
    p.add_argument("--ensemble_size", type=int, default=10, help="Deep Ensemble members per modality (M).")
    p.add_argument(
        "--ensemble_seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Random seeds, one per Deep Ensemble member.",
    )
    # Evidential Regression (uncertainty_method == 'evidential')
    p.add_argument(
        "--evidential_lambda",
        type=float,
        default=0.01,
        help="Weight on the NIG evidence regularizer.",
    )
    p.add_argument(
        "--evidential_uncertainty",
        type=str,
        default="epistemic",
        choices=list(EVIDENTIAL_UNCERTAINTY_TERMS),
        help="Which NIG term drives fusion. 'epistemic' matches what MC Dropout and "
             "Deep Ensemble supply, keeping the methods comparable.",
    )

    # CUE pretrain (modality MLPs)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_every", type=int, default=1)

    # CUE modalities - the actual set of modalities trained/fused (not just an ablation mask)
    p.add_argument("--active_modalities", type=str, nargs="+", default=list(default_modalities),
                    choices=ALL_MODALITIES, help="Modalities to train + fuse")
    p.add_argument("--gnn_backend", type=str, default="mgssl", choices=list(SUPPORTED_GNN_BACKENDS),
                    help="Which pretrained GNN backs the 'gnn' modality when it's active (MolE or MGSSL).")

    # Downstream MLP regressor/classifier head
    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)
    p.add_argument("--reg_num_epochs", type=int, default=500)
    p.add_argument("--reg_learning_rate", type=float, default=1e-3)
    p.add_argument("--reg_batch_size", type=int, default=256)
    p.add_argument("--reg_save_every", type=int, default=1)
    # Seeded immediately before the head is built, so its initialization does not depend on how
    # much of the global RNG stream the upstream stages (modality MLP pretraining, ensemble
    # members, uncertainty estimation, fusion) happened to consume. Without this the head's init
    # silently varies with the number of active modalities; a single unlucky draw then sends all
    # of a cell's folds into a dead-ReLU constant-output collapse. Keep one value for every
    # dataset / UQ method / modality combination / fold - never tune it per cell.
    p.add_argument("--head_seed", type=int, default=1,
                   help="Seed for the downstream head's weight initialization (default 1).")


def run_cue(args: argparse.Namespace) -> None:
    """
    Args:
        args (argparse.Namespace): parsed CLI args (see add_common_args)

    Returns:
        None
    """
    import time
    start_time = time.time()
    print(f"[INFO] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("=" * 80)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    modalities: List[str] = list(args.active_modalities)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)
    print(f"[INFO] Config saved to {config_path}")

    print("=" * 80)
    print("[START] CUE training")
    print(f"  data_dir    : {args.data_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  device      : {device}")
    print(f"  seed        : {args.seed}")
    print(f"  task_type   : {args.task_type}")
    print(f"  modalities  : {modalities}")
    print(f"  uncertainty : {args.uncertainty_method}")
    if "gnn" in modalities:
        print(f"  gnn_backend : {args.gnn_backend}")
    print("=" * 80)

    all_predictions: List[pd.DataFrame] = []
    collapse_flags: List[Tuple[str, bool, float]] = []

    # --------------------------------------------------
    # Fold discovery
    # --------------------------------------------------
    fold_dirs = sorted(
        d for d in os.listdir(args.data_dir)
        if re.fullmatch(r"fold\d+", d) and os.path.isdir(os.path.join(args.data_dir, d))
    )
    print(f"[INFO] Found folds: {fold_dirs}")

    for fold in tqdm.tqdm(fold_dirs, desc="CUE folds", leave=True, dynamic_ncols=True):
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
        # CUE pretrain (normalized y)
        # --------------------------------------------------
        print("[CUE] Pretraining modality MLPs")
        print(f"  modalities : {modalities}")
        print(f"  hidden_dim : {args.hidden_dim}")
        print(f"  epochs     : {args.num_epochs}")
        print(f"  lr         : {args.learning_rate}")
        print(f"  batch_size : {args.batch_size}")

        cue_cfg = CUEConfigs(
            fold_dir=fold_out,
            device=device,
            modalities=modalities,
            data_fold_dir=fold_path,
            gnn_backend=args.gnn_backend,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            save_every=args.save_every,
            task_type=args.task_type,
            uncertainty_method=args.uncertainty_method,
            mc_samples=args.mc_samples,
            dropout_rate=args.dropout_rate,
            ensemble_size=args.ensemble_size,
            ensemble_seeds=args.ensemble_seeds,
            evidential_lambda=args.evidential_lambda,
            evidential_uncertainty=args.evidential_uncertainty,
        )

        cue_model = CUE(cue_cfg)

        print("[CUE] Pretrain START")
        cue_model.fit(
            tr_id=tr_id,
            va_id=va_id,
            X_tr=X_tr,
            X_va=X_va,
            y_tr=y_tr_n.reshape(-1, 1),
            y_va=y_va_n.reshape(-1, 1),
        )
        print("\n" + "=" * 80)
        print("[CUE] Pretrain DONE")

        cue_model.save()
        print("\n" + "=" * 80)
        print("[CUE] Models saved")
        print("=" * 80)

        traj_root = os.path.join(fold_out, "trajectories")

        # --------------------------------------------------
        # Uncertainty computation: LTAU (descriptor/KNN pipeline) vs. MC Dropout/Deep
        # Ensemble (direct predictive-variance pipeline, cue_model.compute_uncertainty)
        # --------------------------------------------------
        estimators: Dict[str, Any] = {}

        if args.uncertainty_method == "ltau":
            print("[LTAU] Extracting modality descriptors")

            D_tr = cue_model.get_modality_descriptors(X_tr)
            D_va = cue_model.get_modality_descriptors(X_va)
            D_te = cue_model.get_modality_descriptors(X_te)

            print("[LTAU] Fitting uncertainty estimators (TRAIN)")

            u_tr = {}
            for m in modalities:
                est_m, df_u_tr_m = fit_ltau_estimator_from_train_trajectories(
                    train_csv=os.path.join(traj_root, m, "train_trajectories.csv"),
                    train_descriptors=D_tr[m],
                    num_bins=args.ltau_bins,
                    k=args.ltau_k,
                    save_dir=os.path.join(fold_out, "ltau", m),
                    uncertainty_mode="entropy",
                )
                estimators[m] = est_m
                u_tr[m] = df_u_tr_m["uncertainty"].values

            print("[LTAU] Estimating VAL uncertainty")
            u_va = {}
            for m in modalities:
                df_u_va_m = estimate_uncertainty_for_split(
                    estimator=estimators[m],
                    split_ids=va_id,
                    split_descriptors=D_va[m],
                    save_path=os.path.join(fold_out, "ltau", m, "val_uncertainty.csv"),
                )
                u_va[m] = df_u_va_m["uncertainty"].values
        else:
            print(f"[Uncertainty] Computing per-modality uncertainty via '{args.uncertainty_method}'")
            u_tr = cue_model.compute_uncertainty(X_tr)
            u_va = cue_model.compute_uncertainty(X_va)

        print(f"[Uncertainty] Train stats ({args.uncertainty_method})")
        for m in modalities:
            print(f"  {m:<4s}: mean={u_tr[m].mean():.4f}, std={u_tr[m].std():.4f}")

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
            custom_weight=args.custom_weight,
        )

        df_fusion_tr = pd.DataFrame({"compound_chembl_id": tr_id})
        for i, m in enumerate(modalities):
            df_fusion_tr[f"u_{m}"] = u_tr[m]
            df_fusion_tr[f"w_{m}"] = W_tr[:, i]
            df_fusion_tr[f"loss_{m}"] = loss_tr[f"loss_{m}"]
        df_fusion_tr.to_csv(os.path.join(fold_out, "fusion", "train_fusion.csv"), index=False)

        np.save(
            os.path.join(fold_out, "embeddings", "train_embeddings.npy"),
            {"compound_chembl_id": tr_id, "emb": H_tr, args.target_col: y_tr},
        )

        # --------------------------------------------
        # VAL embedding + fusion
        # --------------------------------------------
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
            custom_weight=args.custom_weight,
        )

        df_fusion_va = pd.DataFrame({"compound_chembl_id": va_id})
        for i, m in enumerate(modalities):
            df_fusion_va[f"u_{m}"] = u_va[m]
            df_fusion_va[f"w_{m}"] = W_va[:, i]
            df_fusion_va[f"loss_{m}"] = loss_va[f"loss_{m}"]
        df_fusion_va.to_csv(os.path.join(fold_out, "fusion", "val_fusion.csv"), index=False)

        np.save(
            os.path.join(fold_out, "embeddings", "val_embeddings.npy"),
            {"compound_chembl_id": va_id, "emb": H_va, args.target_col: y_va},
        )

        # --------------------------------------------------
        # TEST uncertainty
        # --------------------------------------------------
        if args.uncertainty_method == "ltau":
            print("[LTAU] Estimating TEST uncertainty")
            u_te = {}
            for m in modalities:
                df_u_te_m = estimate_uncertainty_for_split(
                    estimator=estimators[m],
                    split_ids=te_id,
                    split_descriptors=D_te[m],
                    save_path=os.path.join(fold_out, "ltau", m, "test_uncertainty.csv"),
                )
                u_te[m] = df_u_te_m["uncertainty"].values
        else:
            u_te = cue_model.compute_uncertainty(X_te)

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
            custom_weight=args.custom_weight,
        )

        df_fusion_te = pd.DataFrame({"compound_chembl_id": te_id})
        for i, m in enumerate(modalities):
            df_fusion_te[f"u_{m}"] = u_te[m]
            df_fusion_te[f"w_{m}"] = W_te[:, i]
            df_fusion_te[f"loss_{m}"] = loss_te[f"loss_{m}"]
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
            seed=args.head_seed,
        )

        # Re-seed here so the head's init is independent of the upstream RNG consumption.
        seed(int(head_cfg.seed))
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

        # Degenerate-head check. A head whose ReLU units have all died outputs the final layer's
        # bias for every molecule, so the predictions carry zero variance. Training does not
        # error, the loss simply flattens, so without this the failure reaches the result tables
        # unnoticed. Recorded per fold as well as printed.
        pred_std = float(df_fold["mlp"].std())
        collapsed = pred_std < 1e-7
        if collapsed:
            print(f"[{fold}] [COLLAPSE] the head predicts a constant "
                  f"({df_fold['mlp'].iloc[0]:.6f}) for all {len(df_fold)} test molecules "
                  f"(pred std {pred_std:.3e}); this fold's metrics are a training failure, "
                  f"not model performance.", flush=True)
        with open(os.path.join(fold_out, "head_check.json"), "w") as f:
            json.dump({"fold": fold, "head_seed": int(head_cfg.seed),
                       "test_pred_std": pred_std, "collapsed": collapsed,
                       "n_test": int(len(df_fold))}, f, indent=2)
        collapse_flags.append((fold, collapsed, pred_std))

        print(f"[{fold}] DONE")

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    print("=" * 80)
    print("[FINAL] Aggregating metrics")

    n_collapsed = sum(1 for _, c, _ in collapse_flags if c)
    if n_collapsed:
        print(f"[FINAL] [COLLAPSE] {n_collapsed}/{len(collapse_flags)} fold(s) produced constant "
              f"predictions: {[f for f, c, _ in collapse_flags if c]}. The metrics below are a "
              f"training failure for those folds - do not report them as performance.", flush=True)
    with open(os.path.join(args.output_dir, "head_check.json"), "w") as f:
        json.dump({"head_seed": int(args.head_seed),
                   "n_folds": len(collapse_flags), "n_collapsed": n_collapsed,
                   "per_fold": [{"fold": fo, "collapsed": c, "test_pred_std": s}
                                for fo, c, s in collapse_flags]}, f, indent=2)

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

    print_elapsed_time(start_time, time.time())
