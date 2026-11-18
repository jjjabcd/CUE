"""
Speed benchmark for cue4unc (MC Dropout / Deep Ensemble uncertainty) prediction.

Mirrors predict.py's per-batch inference cost exactly (compute_uncertainty,
fusion, MLP head, inverse transform) but only supports a single target_col +
single fold, records cumulative/chunk wall-clock time every
--checkpoint_interval molecules instead of writing prediction values, and
first runs a sanity check confirming the ensemble/dropout uncertainty is
actually being computed with the configured number of members/samples.
"""
import re
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from cue.core.model import CUE, CUEConfigs
from cue.core.downstream_mlp import DownstreamMLP
from cue.core.trainer import DownstreamTrainer
from cue.data.normalization import LabelNormalizer
from cue.utils.utils import load_fold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="./results/MC_Dropout",
                    help="e.g. ./results/MC_Dropout or ./results/Deep_Ensemble")
    p.add_argument("--data_name", type=str, default="DIS")
    p.add_argument("--target_cols", type=str, nargs="+", default=["PI"])
    p.add_argument("--version", type=str, default="v1")
    p.add_argument("--model_name", type=str, default="CUE")
    p.add_argument("--input_file", type=str, required=True, help="CSV file containing SMILES column")
    p.add_argument("--smiles_col", type=str, default="SMILES")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--specific_fold", type=str, default="fold1")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--fusion_policy", type=str, default="softmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--uncertainty_norm", type=str, default="none")

    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)

    p.add_argument("--checkpoint_interval", type=int, default=10_000)
    p.add_argument("--max_molecules", type=int, default=None, help="If set, only use the first N molecules from --input_file")
    p.add_argument("--output_csv", type=str, default="./results/benchmark/timing.csv")
    p.add_argument("--verify_n", type=int, default=100, help="Sample size used for the uncertainty sanity check")
    p.add_argument("--skip_verify", action="store_true")
    return p.parse_args()


def verify_uncertainty(cue_model, sample_smiles):
    """Confirm the configured number of ensemble members / MC-dropout samples
    are actually exercised inside compute_uncertainty(), and that they
    produce non-degenerate (non-identical) outputs."""
    method = cue_model.cfg.uncertainty_method
    print("=" * 60)
    print(f"[Verify] uncertainty_method = {method}")
    ok = True

    if method == "deep_ensemble":
        expected = cue_model.cfg.ensemble_size
        n_fp = len(cue_model.mlp_fp)
        n_img = len(cue_model.mlp_img)
        print(f"[Verify] loaded members: fp={n_fp}, img={n_img} (expected {expected})")
        ok &= (n_fp == expected and n_img == expected)

        w0 = next(cue_model.mlp_fp[0].model.parameters())
        w1 = next(cue_model.mlp_fp[1].model.parameters())
        distinct = not torch.equal(w0, w1)
        print(f"[Verify] member0 vs member1 fp weights distinct: {distinct}")
        ok &= distinct

        call_counts = {"fp": 0, "img": 0}

        def make_hook(key):
            def hook(module, inp, out):
                call_counts[key] += 1
            return hook

        handles = [t.model.register_forward_hook(make_hook("fp")) for t in cue_model.mlp_fp]
        handles += [t.model.register_forward_hook(make_hook("img")) for t in cue_model.mlp_img]

        _ = cue_model.compute_uncertainty(np.array(sample_smiles))

        for h in handles:
            h.remove()

        print(f"[Verify] forward calls during compute_uncertainty(): fp={call_counts['fp']}, img={call_counts['img']} (expected {expected} each)")
        ok &= (call_counts["fp"] == expected and call_counts["img"] == expected)

    elif method == "mc_dropout":
        expected = cue_model.cfg.mc_samples
        call_counts = {"fp": 0, "img": 0}
        outputs = {"fp": [], "img": []}

        def make_hook(key):
            def hook(module, inp, out):
                call_counts[key] += 1
                outputs[key].append(out.detach().cpu().clone())
            return hook

        h_fp = cue_model.mlp_fp[0].model.register_forward_hook(make_hook("fp"))
        h_img = cue_model.mlp_img[0].model.register_forward_hook(make_hook("img"))

        _ = cue_model.compute_uncertainty(np.array(sample_smiles))

        h_fp.remove()
        h_img.remove()

        print(f"[Verify] forward calls during compute_uncertainty(): fp={call_counts['fp']}, img={call_counts['img']} (expected {expected} each)")
        ok &= (call_counts["fp"] == expected and call_counts["img"] == expected)

        var_fp = torch.stack(outputs["fp"], dim=0).var(dim=0).mean().item() if outputs["fp"] else 0.0
        var_img = torch.stack(outputs["img"], dim=0).var(dim=0).mean().item() if outputs["img"] else 0.0
        print(f"[Verify] output variance across {expected} dropout passes: fp={var_fp:.6g}, img={var_img:.6g} (should be > 0)")
        ok &= (var_fp > 0 and var_img > 0)

    else:
        print(f"[Verify] Unknown uncertainty_method={method!r}; skipping checks")
        ok = False

    print(f"[Verify] Result: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return ok


def timed_compute_uncertainty(cue_model, batch_smiles):
    """
    Calls compute_uncertainty() but isolates and returns the wall-clock time
    spent strictly inside the modality MLPs' forward() calls (the mc_dropout
    T-sample loop or the deep_ensemble member loop), excluding fp/img feature
    extraction which also happens inside compute_uncertainty(). Hooks are
    installed/removed around this single call so forward passes elsewhere
    (e.g. predict()'s embedding extraction, which reuses the same modules)
    are never counted.
    """
    state = {"accum": 0.0, "start": None}

    def pre_hook(module, inp):
        state["start"] = time.perf_counter()

    def post_hook(module, inp, out):
        if state["start"] is not None:
            state["accum"] += time.perf_counter() - state["start"]
            state["start"] = None

    handles = []
    for t in cue_model.mlp_fp:
        handles.append(t.model.register_forward_pre_hook(pre_hook))
        handles.append(t.model.register_forward_hook(post_hook))
    for t in cue_model.mlp_img:
        handles.append(t.model.register_forward_pre_hook(pre_hook))
        handles.append(t.model.register_forward_hook(post_hook))

    uncertainties = cue_model.compute_uncertainty(np.array(batch_smiles))

    for h in handles:
        h.remove()

    return uncertainties, state["accum"]


def main():
    args = parse_args()
    if len(args.target_cols) != 1:
        raise ValueError("time_predict.py supports exactly one --target_cols value")
    if not args.specific_fold:
        raise ValueError("time_predict.py requires --specific_fold (e.g. fold1)")

    device = torch.device(args.device)
    target_col = args.target_cols[0]

    try:
        df_in = pd.read_csv(args.input_file, engine="python")
    except Exception:
        df_in = pd.read_csv(args.input_file)
    if args.smiles_col not in df_in.columns:
        raise ValueError(f"Column {args.smiles_col} not found in {args.input_file}")
    smiles_list = df_in[args.smiles_col].tolist()
    if args.max_molecules is not None:
        smiles_list = smiles_list[:args.max_molecules]
    print(f"[Input] {len(smiles_list)} SMILES loaded from {args.input_file}")

    task_type = "classification" if "_label" in target_col else "regression"
    model_root = os.path.join(args.base_dir, args.data_name, target_col, args.version, args.model_name)
    if not os.path.exists(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")

    fold_dirs = sorted(
        d for d in os.listdir(model_root)
        if re.fullmatch(r"fold\d+", d) and os.path.isdir(os.path.join(model_root, d))
    )
    if args.specific_fold not in fold_dirs:
        raise FileNotFoundError(f"Specific fold {args.specific_fold} not found in {model_root}")
    fold_path = os.path.join(model_root, args.specific_fold)
    print(f"[Model] {model_root} | fold={args.specific_fold} | task={task_type}")

    config_path = os.path.join(model_root, "config.json")
    train_cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_cfg = json.load(f)

    cue_cfg = CUEConfigs(
        fold_dir=fold_path,
        device=device,
        hidden_dim=args.hidden_dim,
        task_type=task_type,
        batch_size=args.batch_size,
        uncertainty_method=train_cfg.get("uncertainty_method", "mc_dropout"),
        mc_samples=train_cfg.get("mc_samples", 10),
        dropout_rate=train_cfg.get("dropout_rate", 0.1),
        ensemble_size=train_cfg.get("ensemble_size", 10),
        ensemble_seeds=train_cfg.get("ensemble_seeds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    )
    cue_model = CUE(cue_cfg)
    cue_model.load()

    if not args.skip_verify:
        sample_smiles = smiles_list[:min(args.verify_n, len(smiles_list))]
        if not verify_uncertainty(cue_model, sample_smiles):
            raise RuntimeError("Uncertainty sanity check FAILED — aborting before timing benchmark.")

    active_mods = ["fp", "img"] if args.model_name == "CUE" else args.model_name.split("_")

    head_dir = os.path.join(fold_path, "regressors" if task_type == "regression" else "classifiers")
    mlp_path = os.path.join(head_dir, "MLP.pth")
    head_model = DownstreamMLP(
        input_dim=int(args.hidden_dim),
        hidden_dim_1=args.reg_hidden_dim_1,
        hidden_dim_2=args.reg_hidden_dim_2,
        hidden_dim_3=args.reg_hidden_dim_3,
        output_dim=1,
    )
    trainer = DownstreamTrainer(model=head_model, device=device)
    trainer.load_model(mlp_path)

    lab_path = os.path.join(head_dir, "label_normalizer.pkl")
    lab = None
    if os.path.exists(lab_path):
        lab = LabelNormalizer.load(lab_path)
    else:
        try:
            _, _, _, _, y_tr_refit, _, _, _, _ = load_fold(fold_path, target_col)
            lab_mode = train_cfg.get("label_mode", "zscore")
            if task_type == "classification":
                lab_mode = "none"
            lab = LabelNormalizer(lab_mode)
            lab.fit(y_tr_refit)
        except Exception:
            lab = None

    # --------------------------------------------------
    # Timed batch inference
    # --------------------------------------------------
    num_batches = int(np.ceil(len(smiles_list) / args.batch_size))
    n_seen = 0
    timing_records = []
    next_checkpoint = args.checkpoint_interval
    t_start = time.perf_counter()
    last_checkpoint_time = t_start
    uncertainty_time_accum = 0.0  # wall-clock spent only in the MC-Dropout/Ensemble forward passes (excludes feature extraction)

    for b_idx in tqdm(range(num_batches), desc=f"[{cue_cfg.uncertainty_method}] Predicting"):
        start = b_idx * args.batch_size
        end = min((b_idx + 1) * args.batch_size, len(smiles_list))
        batch_smiles = smiles_list[start:end]

        uncertainties, batch_uncertainty_time = timed_compute_uncertainty(cue_model, batch_smiles)
        uncertainty_time_accum += batch_uncertainty_time
        for m in ["fp", "img"]:
            if m not in active_mods:
                uncertainties[m] = np.full_like(uncertainties[m], 1e9)

        emb, W = cue_model.predict(
            np.array(batch_smiles),
            uncertainties=uncertainties,
            fusion_policy=args.fusion_policy,
            temperature=args.temperature,
            uncertainty_norm=args.uncertainty_norm,
            task_type=task_type,
            return_weights=True,
        )

        y_pred_z = trainer.inference(emb, batch_size=args.batch_size, task_type=task_type, return_logits=True)
        if task_type == "classification":
            _ = 1.0 / (1.0 + np.exp(-y_pred_z))
        elif lab is not None:
            _ = lab.inverse_transform(y_pred_z)

        n_seen += len(batch_smiles)
        while n_seen >= next_checkpoint:
            now = time.perf_counter()
            timing_records.append({
                "n_processed": next_checkpoint,
                "elapsed_sec": now - t_start,
                "chunk_sec": now - last_checkpoint_time,
                "uncertainty_elapsed_sec": uncertainty_time_accum,
            })
            last_checkpoint_time = now
            next_checkpoint += args.checkpoint_interval

    if not timing_records or timing_records[-1]["n_processed"] != n_seen:
        now = time.perf_counter()
        timing_records.append({
            "n_processed": n_seen,
            "elapsed_sec": now - t_start,
            "chunk_sec": now - last_checkpoint_time,
            "uncertainty_elapsed_sec": uncertainty_time_accum,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    timing_df = pd.DataFrame(timing_records)
    timing_df.to_csv(args.output_csv, index=False)
    total_elapsed = timing_records[-1]["elapsed_sec"]
    print(f"[SUCCESS] {cue_cfg.uncertainty_method} | {n_seen} molecules | {total_elapsed:.2f}s total | timing saved to {args.output_csv}")


if __name__ == "__main__":
    main()
