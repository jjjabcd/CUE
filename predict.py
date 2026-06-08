import re
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

from tqdm import tqdm
from cue.core.model import CUE, CUEConfigs
from cue.core.downstream_mlp import DownstreamMLP, DownstreamMLPConfigs
from cue.ltau.estimator import LTAUEstimator
from cue.core.trainer import DownstreamTrainer
from cue.utils.utils import LabelNormalizer, load_fold
from cue.utils.seed import seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="./results", help="Base results directory")
    p.add_argument("--data_name", type=str, default="DIS", help="Dataset name e.g. DIS, INH")
    p.add_argument("--target_cols", type=str, nargs="+", default=None, help="Target column names (defines the folder structure)")
    p.add_argument("--target_col", type=str, default=None, help="Target column name (singular alias)")
    p.add_argument("--version", type=str, default="v1", help="Experiment version (or ablation version)")
    p.add_argument("--model_name", type=str, default="CUE", help="Subdirectory name usually CUE or ablation variant like 'fp_img'")
    p.add_argument("--input_smiles", type=str, nargs="+", help="One or more SMILES strings to predict")
    p.add_argument("--input_file", type=str, help="CSV file containing SMILES column")
    p.add_argument("--smiles_col", type=str, default="SMILES", help="SMILES column name in input file")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--specific_fold", type=str, default=None, help="If specified (e.g. 'fold1'), run prediction only using this fold")
    p.add_argument("--device", type=str, default="cuda:0")
    
    # Needs to match training config if not loaded from config file
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    
    # MLP Config
    p.add_argument("--reg_hidden_dim_1", type=int, default=16)
    p.add_argument("--reg_hidden_dim_2", type=int, default=16)
    p.add_argument("--reg_hidden_dim_3", type=int, default=8)
    
    return p.parse_args()

def load_or_reconstruct_estimator(base_path: str, mod: str, k: int = 5) -> LTAUEstimator:
    """
    Load estimator.pkl if exists, else reconstruct from train_*.npy files.
    """
    ltau_dir = os.path.join(base_path, "ltau", mod)
    pkl_path = os.path.join(ltau_dir, "estimator.pkl")
    
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    
    # Check for npy files
    desc_path = os.path.join(ltau_dir, "train_descriptors.npy")
    pdfs_path = os.path.join(ltau_dir, "train_pdfs.npy")
    bins_path = os.path.join(ltau_dir, "train_bin_edges.npy")
    
    if os.path.exists(desc_path) and os.path.exists(pdfs_path) and os.path.exists(bins_path):
        print(f"[LTAU] Reconstructing {mod} estimator from npy files (k={k})")
        ref_emb = np.load(desc_path)
        reference_pdfs = np.load(pdfs_path)
        shared_bin_edges = np.load(bins_path)

        est = LTAUEstimator(k=k)
        est.fit(ref_emb=ref_emb, reference_pdfs=reference_pdfs, shared_bin_edges=shared_bin_edges)
        return est
    
    raise FileNotFoundError(f"Could not find estimator.pkl or train_*.npy in {ltau_dir}")


def predict_on_smiles(args):
    device = torch.device(args.device)
    
    # Handle Input SMILES
    if args.input_smiles:
        smiles_list = args.input_smiles
        ids = [f"Input_{i}" for i in range(len(smiles_list))]
    elif args.input_file:
        try:
            df_in = pd.read_csv(args.input_file, engine="python")
        except:
            df_in = pd.read_csv(args.input_file)
        if args.smiles_col not in df_in.columns:
            raise ValueError(f"Column {args.smiles_col} not found in {args.input_file}")
        smiles_list = df_in[args.smiles_col].tolist()
        if "Compound" in df_in.columns:
             ids = df_in["Compound"].tolist()
        elif "ID" in df_in.columns:
             ids = df_in["ID"].tolist()
        else:
             ids = [f"CID_{i}" for i in range(len(smiles_list))]
    else:
        raise ValueError("Must provide --input_smiles or --input_file")

    print(f"[Input] {len(smiles_list)} SMILES loaded.")
    
    results_dict = {
        "ID": ids,
        "SMILES": smiles_list
    }

    # Handle target_cols and target_col alias
    if not args.target_cols and args.target_col:
        args.target_cols = [args.target_col]
    if not args.target_cols:
        raise ValueError("Must provide either --target_cols or --target_col")

    # Iterate over Target Cols
    for target_col in args.target_cols:
        print("="*60)
        print(f"[Target] Processing {target_col}...")
        
        # Re-construct model_root for THIS target
        model_root = os.path.join(args.base_dir, args.data_name, target_col, args.version, args.model_name)
    
        if not os.path.exists(model_root):
            print(f"[WARN] Model directory not found: {model_root}. Skipping {target_col}.")
            continue
            
        fold_dirs = sorted(
            d for d in os.listdir(model_root)
            if re.fullmatch(r"fold\d+", d) and os.path.isdir(os.path.join(model_root, d))
        )
        
        if args.specific_fold:
            if args.specific_fold in fold_dirs:
                fold_dirs = [args.specific_fold]
                print(f"[Models] Using specific fold: {args.specific_fold}")
            else:
                print(f"[WARN] Specific fold {args.specific_fold} not found in {model_root}. Skipping.")
                continue
        
        if not fold_dirs:
            print(f"[WARN] No fold directories found in {model_root}. Skipping.")
            continue
            
        print(f"[Models] Found {len(fold_dirs)} folds in {model_root}")

        # Prediction Loop (Batched)
        target_fold_preds = []
        target_fold_u_fp = []
        target_fold_u_img = []
        target_fold_w_fp = []
        target_fold_w_img = []
        
        for fold in fold_dirs:
            fold_path = os.path.join(model_root, fold)
            
            # 1. Load CUE
            cue_cfg = CUEConfigs(
                fold_dir=fold_path,
                device=device,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size
            )
            cue_model = CUE(cue_cfg)
            cue_model.load()
            
            # Load Config to get LTAU K
            config_path = os.path.join(model_root, "config.json")
            ltau_k = 5 
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    ltau_k = cfg.get("ltau_k", 5)

            # Load LTAU estimators
            est_fp = load_or_reconstruct_estimator(fold_path, "fp", k=ltau_k)
            est_img = load_or_reconstruct_estimator(fold_path, "img", k=ltau_k)
            
            # Load MLP
            head_dir = os.path.join(fold_path, "regressors")
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

            # Label Normalizer
            lab_path = os.path.join(head_dir, "label_normalizer.pkl")
            lab = None
            if os.path.exists(lab_path):
                lab = LabelNormalizer.load(lab_path)
            else:
                try:
                    _, _, _, _, y_tr_refit, _, _, _, _ = load_fold(fold_path, target_col)
                    config_path = os.path.join(model_root, "config.json")
                    lab_mode = "zscore"
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                             lab_mode = json.load(f).get("label_mode", "zscore")
                    lab = LabelNormalizer(lab_mode)
                    lab.fit(y_tr_refit)
                except Exception:
                    lab = None

            # Batch Inference
            fold_preds_flat = []
            fold_u_fp_flat = []
            fold_u_img_flat = []
            fold_w_fp_flat = []
            fold_w_img_flat = []
            from cue.ltau.pipeline import estimate_uncertainty_for_split

            num_batches = int(np.ceil(len(smiles_list) / args.batch_size))
            
            for b_idx in tqdm(range(num_batches), desc=f"[{fold}] Predicting", leave=False):
                start = b_idx * args.batch_size
                end = min((b_idx + 1) * args.batch_size, len(smiles_list))
                batch_smiles = smiles_list[start:end]
                
                # 2. Extract Descriptors
                D = cue_model.get_modality_descriptors(np.array(batch_smiles))
                
                # Estimate Uncertainty
                dummy_ids = np.zeros(len(batch_smiles))
                
                df_u_fp = estimate_uncertainty_for_split(est_fp, dummy_ids, D["fp"], save_path=None)
                u_fp = df_u_fp["uncertainty"].values
                
                df_u_img = estimate_uncertainty_for_split(est_img, dummy_ids, D["img"], save_path=None)
                u_img = df_u_img["uncertainty"].values
                
                uncertainties = {"fp": u_fp, "img": u_img}
                
                # 3. Get Embeddings
                emb, W = cue_model.predict(
                    np.array(batch_smiles),
                    uncertainties=uncertainties,
                    temperature=args.temperature,
                    return_weights=True
                )
                
                # 4. MLP Inference
                y_pred_z = trainer.inference(emb, batch_size=args.batch_size)
                
                # 5. Inverse Transform
                if lab is not None:
                    y_final = lab.inverse_transform(y_pred_z)
                else:
                    y_final = y_pred_z
                
                fold_preds_flat.extend(y_final.reshape(-1).tolist())
                fold_u_fp_flat.extend(u_fp.tolist())
                fold_u_img_flat.extend(u_img.tolist())
                fold_w_fp_flat.extend(W[:, 0].tolist())
                fold_w_img_flat.extend(W[:, 1].tolist())
            
            # End Batch Loop (Fold)
            target_fold_preds.append(fold_preds_flat)
            target_fold_u_fp.append(fold_u_fp_flat)
            target_fold_u_img.append(fold_u_img_flat)
            target_fold_w_fp.append(fold_w_fp_flat)
            target_fold_w_img.append(fold_w_img_flat)
            
            # Clean up explicit large objects
            del cue_model, head_model, trainer, D, emb
            torch.cuda.empty_cache()

        # Aggregate Folds
        if target_fold_preds:
            target_fold_preds = np.array(target_fold_preds)
            y_mean = np.mean(target_fold_preds, axis=0)
            results_dict[target_col] = y_mean
            results_dict["u_fp"] = np.mean(target_fold_u_fp, axis=0)
            results_dict["u_img"] = np.mean(target_fold_u_img, axis=0)
            results_dict["w_fp"] = np.mean(target_fold_w_fp, axis=0)
            results_dict["w_img"] = np.mean(target_fold_w_img, axis=0)
            
    # Save Results
    df_out = pd.DataFrame(results_dict)
    
    # Reorder columns: ID, u_fp, u_img, w_fp, w_img, targets, SMILES
    cols_order = ["ID", "u_fp", "u_img", "w_fp", "w_img"]
    for t in args.target_cols:
        if t in df_out.columns:
            cols_order.append(t)
    cols_order.append("SMILES")
    
    # Keep only existing columns to avoid errors
    final_cols = [c for c in cols_order if c in df_out.columns]
    df_out = df_out[final_cols]
    
    print("\n[Results Preview]")
    print(df_out.head())
    
    out_name = f"prediction_{args.data_name}_{args.model_name}.csv"
    if len(args.target_cols) == 1:
        out_name = f"prediction_{args.data_name}_{args.target_cols[0]}_{args.version}_{args.model_name}.csv"
        
    df_out.to_csv(out_name, index=False)
    print(f"\nSaved to {out_name}")

if __name__ == "__main__":
    args = parse_args()
    predict_on_smiles(args)
