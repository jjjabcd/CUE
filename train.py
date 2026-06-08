from __future__ import annotations

import re
import time
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import torch
import tqdm

from cue.core.model import CUE, CUEConfigs
from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs
from cue.core.trainer import DownstreamTrainer
from cue.utils.utils import LabelNormalizer, ensure_dirs, load_fold, aggregate_metrics, print_elapsed_time
from cue.utils.seed import seed
from cue.utils.metric import get_metrics

from cue.ltau.pipeline import (
    fit_ltau_estimator_from_train_trajectories,
    estimate_uncertainty_for_split,
)

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*Asking to truncate to max_length.*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--target_col", type=str, default="Ssel")
    p.add_argument("--label_mode", type=str, default="zscore")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    # Fusion
    p.add_argument("--temperature", type=float, default=1.0)

    # LTAU
    p.add_argument("--ltau_bins", type=int, default=5000)
    p.add_argument("--ltau_k", type=int, default=5)

    # CUE pretrain (modality MLPs)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_every", type=int, default=1)

    # Downstream MLP regressor head
    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)
    p.add_argument("--reg_num_epochs", type=int, default=500)
    p.add_argument("--reg_learning_rate", type=float, default=1e-3)
    p.add_argument("--reg_batch_size", type=int, default=256)
    p.add_argument("--reg_save_every", type=int, default=1)

    return p.parse_args()


def main() -> None:
    start_time = time.time()
    print(f"[INFO] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


    args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)
    print(f"[INFO] Config saved to {config_path}")

    print("[START] CUE training")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  device    : {device}")
    print(f"  seed      : {args.seed}")

    all_predictions: List[pd.DataFrame] = []

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
        print(f"[{fold}] START")

        # --------------------------------------------------
        # Fold setup
        # --------------------------------------------------
        fold_path = os.path.join(args.data_dir, fold)
        fold_out = os.path.join(args.output_dir, fold)
        ensure_dirs(fold_out)

        print(f"[{fold}] Loading data")
        tr_id, va_id, te_id, X_tr, y_tr, X_va, y_va, X_te, y_te = load_fold(fold_path, args.target_col)

        print("[Data]")
        print(f"  Train: X={X_tr.shape}, y={y_tr.shape}")
        print(f"  Val  : X={X_va.shape}, y={y_va.shape}")
        print(f"  Test : X={X_te.shape}, y={y_te.shape}")

        # --------------------------------------------------
        # Label normalization (single source of truth)
        # --------------------------------------------------
        lab = LabelNormalizer(args.label_mode)
        lab.fit(y_tr)

        y_tr_n = lab.transform(y_tr)
        y_va_n = lab.transform(y_va)
        y_te_n = lab.transform(y_te)

        # --------------------------------------------------
        # CUE pretrain (normalized y)
        # --------------------------------------------------
        print("[CUE] Pretraining modality MLPs")
        print(f"  hidden_dim : {args.hidden_dim}")
        print(f"  epochs     : {args.num_epochs}")
        print(f"  lr         : {args.learning_rate}")
        print(f"  batch_size : {args.batch_size}")

        cue_cfg = CUEConfigs(
            fold_dir=fold_out,
            device=device,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            save_every=args.save_every,
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
        print("[CUE] Pretrain DONE")

        cue_model.save()
        print("[CUE] Models saved")
        
        traj_root = os.path.join(fold_out, "trajectories")

        # --------------------------------------------------
        # LTAU: descriptor extraction (hidden layer)
        # --------------------------------------------------

        D_tr = cue_model.get_modality_descriptors(X_tr)
        D_va = cue_model.get_modality_descriptors(X_va)
        D_te = cue_model.get_modality_descriptors(X_te)

        # --------------------------------------------------
        # LTAU: fit estimators from TRAIN only
        # --------------------------------------------------
        print("[LTAU] Fitting uncertainty estimators (TRAIN)")

        est_fp, df_u_tr_fp = fit_ltau_estimator_from_train_trajectories(
            train_csv=os.path.join(traj_root, "fp", "train_trajectories.csv"),
            train_descriptors=D_tr["fp"],
            num_bins=args.ltau_bins,
            k=args.ltau_k,
            save_dir=os.path.join(fold_out, "ltau", "fp"),
        )

        est_img, df_u_tr_img = fit_ltau_estimator_from_train_trajectories(
            train_csv=os.path.join(traj_root, "img", "train_trajectories.csv"),
            train_descriptors=D_tr["img"],
            num_bins=args.ltau_bins,
            k=args.ltau_k,
            save_dir=os.path.join(fold_out, "ltau", "img"),
        )

        u_tr = {
            "fp": df_u_tr_fp["uncertainty"].values, 
            "img": df_u_tr_img["uncertainty"].values,
        }

        # --------------------------------------------------
        # VAL uncertainty
        # --------------------------------------------------

        df_u_va_fp = estimate_uncertainty_for_split(
            estimator=est_fp,
            split_ids=va_id,
            split_descriptors=D_va["fp"],
            save_path=os.path.join(fold_out, "ltau", "fp", "val_uncertainty.csv"),
        )

        df_u_va_img = estimate_uncertainty_for_split(
            estimator=est_img,
            split_ids=va_id,
            split_descriptors=D_va["img"],
            save_path=os.path.join(fold_out, "ltau", "img", "val_uncertainty.csv"),
        )
        
        # CUE: Uncertainty-aware fusion is used (no frozen weights/masking)

        # --------------------------------------------------
        # TRAIN embedding + fusion
        # --------------------------------------------------
        print("[Embedding] Generating TRAIN embeddings")

        uni_tr, W_tr, loss_tr = cue_model.predict(
            X_tr,
            y=y_tr_n,
            uncertainties=u_tr,
            temperature=args.temperature,
            return_weights=True,
            return_losses=True,
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
            os.path.join(fold_out, "emb", "train_emb.npy"),
            {"compound_chembl_id": tr_id, "emb": uni_tr, args.target_col: y_tr},
        )
        
        # --------------------------------------------
        # VAL embedding + fusion
        # --------------------------------------------
        print("[Embedding] Generating VAL embeddings")

        uni_va, W_va, loss_va = cue_model.predict(
            X_va,
            y=y_va_n,
            uncertainties={
                "fp": df_u_va_fp["uncertainty"].values,
                "img": df_u_va_img["uncertainty"].values,
            },
            temperature=args.temperature,
            return_weights=True,
            return_losses=True,
        )

        df_fusion_va = pd.DataFrame({
            "compound_chembl_id": va_id,
            "u_fp": df_u_va_fp["uncertainty"].values,
            "u_img": df_u_va_img["uncertainty"].values,
            "w_fp": W_va[:, 0],
            "w_img": W_va[:, 1],
            "loss_fp": loss_va["loss_fp"],
            "loss_img": loss_va["loss_img"],
        })

        df_fusion_va.to_csv(os.path.join(fold_out, "fusion", "va_fusion.csv"), index=False)
        
        np.save(
            os.path.join(fold_out, "emb", "va_emb.npy"),
            {"compound_chembl_id": va_id, "emb": uni_va, args.target_col: y_va},
        )

        # --------------------------------------------------
        # TEST uncertainty
        # --------------------------------------------------

        df_u_te_fp = estimate_uncertainty_for_split(
            estimator=est_fp,
            split_ids=te_id,
            split_descriptors=D_te["fp"],
            save_path=os.path.join(fold_out, "ltau", "fp", "test_uncertainty.csv"),
        )
        df_u_te_img = estimate_uncertainty_for_split(
            estimator=est_img,
            split_ids=te_id,
            split_descriptors=D_te["img"],
            save_path=os.path.join(fold_out, "ltau", "img", "test_uncertainty.csv"),
        )

        u_te = {
            "fp": df_u_te_fp["uncertainty"].values,
            "img": df_u_te_img["uncertainty"].values,
        }

        # --------------------------------------------------
        # TEST embedding + fusion
        # --------------------------------------------------

        uni_te, W_te, loss_te = cue_model.predict(
            X_te,
            y=y_te_n,
            uncertainties=u_te,
            temperature=args.temperature,
            return_weights=True,
            return_losses=True,
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
            os.path.join(fold_out, "emb", "test_emb.npy"),
            {"compound_chembl_id": te_id, "emb": uni_te, args.target_col: y_te},
        )

        # --------------------------------------------------
        # Downstream head (MLP)
        # --------------------------------------------------
        print("[Regressor] Training downstream MLP regressor")
        reg_dir = os.path.join(fold_out, "regressors")
        os.makedirs(reg_dir, exist_ok=True)
        lab.save(os.path.join(reg_dir, "label_normalizer.pkl"))
        mlp_ckpt_path = os.path.join(reg_dir, "MLP.pth")

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
            input_dim=int(uni_tr.shape[1]),
            hidden_dim_1=int(head_cfg.hidden_dim_1),
            hidden_dim_2=int(head_cfg.hidden_dim_2),
            hidden_dim_3=int(head_cfg.hidden_dim_3),
            output_dim=1,
        )

        mlp_trainer = DownstreamTrainer(model=head_model, device=device)

        history = mlp_trainer.fit(
            feature_name="mlp",
            X=uni_tr,
            y=y_tr_n.reshape(-1, 1) if y_tr_n.ndim == 1 else y_tr_n,
            batch_size=int(head_cfg.batch_size),
            num_epochs=int(head_cfg.num_epochs),
            learning_rate=float(head_cfg.learning_rate),
            validation_data={
                "X": uni_va,
                "y": y_va_n.reshape(-1, 1) if y_va_n.ndim == 1 else y_va_n,
            },
            filepath_ckpt=mlp_ckpt_path,
            save_every=int(head_cfg.save_every),
            save_trajectory=True,
            trajectory_dir=os.path.join(reg_dir, "trajectories"),
            compound_ids_tr=tr_id,
            compound_ids_va=va_id,
            smiles_tr=X_tr,
            smiles_va=X_va,
        )

        mlp_trainer.plot(
            history=history,
            save_dir=reg_dir,
            best_key="val_rmse",
        )

        mlp_trainer.load_model(mlp_ckpt_path)

        # --------------------------------------------------
        # Test prediction
        # --------------------------------------------------
        print("[Prediction] Predicting test set")

        z_pred_te_mlp = mlp_trainer.inference(
            uni_te,
            batch_size=1024,
        )

        fold_records: List[Dict[str, Any]] = []
        for i, (cid, gt, h) in enumerate(zip(te_id, y_te, uni_te)):
            row: Dict[str, Any] = {
                "compound_chembl_id": cid,
                "GroundTruth": float(gt),
                "Fold": fold,
            }
            row["mlp"] = float(lab.inverse_transform(float(z_pred_te_mlp[i])))
            fold_records.append(row)

        df_fold = pd.DataFrame(fold_records)
        df_fold.to_csv(os.path.join(fold_out, "predictions.csv"), index=False)
        all_predictions.append(df_fold)

        print(f"[{fold}] DONE")

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    print("[FINAL] Aggregating metrics")

    df_all = pd.concat(all_predictions, ignore_index=True)

    df_metrics = aggregate_metrics(df_all)
    df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    df_all.to_csv(os.path.join(args.output_dir, "predictions_all.csv"), index=False)

    print("[DONE] All folds completed")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
