import os
import json
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import joblib

# umap 패치 적용
from sklearn.utils import check_array as sklearn_check_array
from sklearn.utils import validation

def patched_check_array(*args, **kwargs):
    if 'ensure_all_finite' in kwargs:
        kwargs.pop('ensure_all_finite')
    return sklearn_check_array(*args, **kwargs)

validation.check_array = patched_check_array

import umap.umap_ as umap_module
umap_module.check_array = patched_check_array

import umap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cue.core.model import CUE, CUEConfigs
from cue.utils.utils import DATASET_TO_SOURCE, normalize_columns

def _sanitize_embs(embs):
    embs = np.asarray(embs, dtype=np.float32)
    if np.any(np.isnan(embs)) or np.any(np.isinf(embs)):
        embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    return embs

def doUMAP(embs, y, target_col, random_state):
    embs = _sanitize_embs(embs)
    n_neighbors = min(15, max(2, len(embs) - 1))
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=0.1)
    points = reducer.fit_transform(embs)

    df = pd.DataFrame({"UMAP1": points[:, 0], "UMAP2": points[:, 1], target_col: y})
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_pad, y_pad = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    axis_range = {"x_min": x_min - x_pad, "x_max": x_max + x_pad, "y_min": y_min - y_pad, "y_max": y_max + y_pad}

    return df, axis_range

def load_train_analysis_data(results_dir, dataset, fold_num):
    """
    Loads data for TRAIN analysis.
    Metadata: selectivity_train.csv
    Fusion: train_fusion.csv
    Predictions: predictions_all.csv (for mlp values)
    """
    # the shared ChEMBL tables keep the on-disk assay folder names (DIS -> Kd)
    data_dir = f'../chembl/data/selectivity_df/{DATASET_TO_SOURCE.get(dataset, dataset)}/fold{fold_num}'
    df_meta = normalize_columns(pd.read_csv(os.path.join(data_dir, 'selectivity_train.csv')))
    
    fusion_dir = os.path.join(results_dir, 'fusion')
    df_fusion = pd.read_csv(os.path.join(fusion_dir, 'train_fusion.csv'))
    
    # Merge meta and fusion
    df = pd.merge(df_meta, df_fusion, on='compound_chembl_id', how='left')
    
    # Load Predictions (mlp) - merging by compound_chembl_id
    base_results_dir = os.path.dirname(results_dir)
    pred_path = os.path.join(base_results_dir, 'predictions_all.csv')
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        # Drop duplicates in case a compound appears in multiple folds (though it shouldn't)
        df_pred = df_pred.drop_duplicates(subset=['compound_chembl_id'])
        df = pd.merge(df, df_pred[['compound_chembl_id', 'mlp']], on='compound_chembl_id', how='left')
    
    return df

def plot_uncertainty_umap(df, save_dir, color_col, label, axis_range_umap):
    print(f"  -> Plotting UMAP colored by {color_col}...")
    filepath = os.path.join(save_dir, f'umap_{color_col}.png')
    
    c_vals = df[color_col].values
    mask = ~np.isnan(c_vals) & ~np.isinf(c_vals)
    
    cmap = 'rainbow' if color_col in ['PI', 'Ssel', 'WS2', 'RS2'] else 'viridis'
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(df.loc[mask, 'UMAP1'], df.loc[mask, 'UMAP2'], c=c_vals[mask], cmap=cmap, alpha=0.8, s=20)
    cbar = plt.colorbar(scatter, ax=ax)
    
    # Remove label from colorbar for selectivity (GroundTruth equivalent) as requested
    if color_col in ['PI', 'Ssel', 'WS2', 'RS2']:
        cbar.set_label("", fontsize=12)
    else:
        cbar.set_label(label, fontsize=12)
    
    ax.set_xlim(axis_range_umap['x_min'], axis_range_umap['x_max'])
    ax.set_ylim(axis_range_umap['y_min'], axis_range_umap['y_max'])
    ax.set_xlabel('UMAP1', fontsize=15)
    ax.set_ylabel('UMAP2', fontsize=15)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Clean version
    ax.set_title(""); ax.set_xlabel(""); ax.set_ylabel(""); cbar.set_label("")
    plt.savefig(filepath.replace('.png', '_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ratio_vs_selectivity(df, save_dir, target_col):
    print(f"  -> Plotting Ratio vs {target_col} (PCC calculation)...")
    x, y = df['u_ratio_v2'].values, df[target_col].values
    mask = ~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    
    if len(x_c) < 2: return
    pcc, p_val = stats.pearsonr(x_c, y_c)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_c, y=y_c, scatter_kws={'alpha':0.8, 's':20}, line_kws={'color':'black', 'ls':'--', 'alpha':0.5, 'lw':1}, ci=None)
    plt.xlabel('Ratio [log2(u_fp / u_img)]', fontsize=12)
    plt.ylabel(f'Actual {target_col}', fontsize=12)
    plt.title(f'PCC: {pcc:.4f}, p: {p_val:.2e}', fontsize=14)
    
    filepath = os.path.join(save_dir, f'ratio_vs_{target_col}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.title(""); plt.xlabel(""); plt.ylabel("")
    plt.savefig(filepath.replace('.png', '_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save PCC results
    res_path = os.path.join(save_dir, 'pcc_results.txt')
    with open(res_path, 'w') as f:
        f.write(f"Target: {target_col}\n")
        f.write(f"PCC: {pcc:.4f}\n")
        f.write(f"p-value (scientific): {p_val:.4e}\n")
        f.write(f"p-value (fixed): {p_val:.4f}\n")
    
    # Save the plot data to CSV as requested
    df_csv = df[mask].copy()
    df_csv = df_csv.rename(columns={'u_ratio_v2': 'ratio'})
    csv_path = os.path.join(save_dir, f'ratio_data_{target_col}.csv')
    df_csv[['compound_chembl_id', 'u_fp', 'u_img', 'ratio', target_col, 'SMILES']].to_csv(csv_path, index=False)
    print(f"  -> Plot data saved to {csv_path}")

def plot_prediction_kde(df, save_dir, target_col):
    if 'mlp' not in df.columns or df['mlp'].isna().all(): return
    print(f"  -> Plotting Prediction KDE (Ratio Threshold -1)...")
    threshold = -1.0
    group_low = df[df['u_ratio_v2'] <= threshold]['mlp']
    group_high = df[df['u_ratio_v2'] > threshold]['mlp']
    
    group_low = group_low[~group_low.isna()]
    group_high = group_high[~group_high.isna()]
    
    if len(group_low) < 2 or len(group_high) < 2: return

    t_stat, p_val_two = stats.ttest_ind(group_low, group_high, equal_var=False, alternative='two-sided')
    _, p_val_one = stats.ttest_ind(group_low, group_high, equal_var=False, alternative='less')

    plt.figure(figsize=(8, 6))
    sns.kdeplot(group_low, label=f'Ratio <= {threshold}', color='blue', alpha=0.4, fill=True)
    sns.kdeplot(group_high, label=f'Ratio > {threshold}', color='red', alpha=0.4, fill=True)
    plt.xlabel(f'{target_col}', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.title(f't-test p-value (two-sided): {p_val_two:.4e} | (one-sided, less): {p_val_one:.4e}', fontsize=12)
    plt.legend()
    
    filepath = os.path.join(save_dir, f'prediction_kde.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.title(""); plt.xlabel(""); plt.ylabel(""); plt.legend().remove()
    plt.savefig(filepath.replace('.png', '_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save t-test results
    res_path = os.path.join(save_dir, 'ttest_results.txt')
    with open(res_path, 'w') as f:
        f.write(f"Target: {target_col}\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value (two-sided, scientific): {p_val_two:.4e}\n")
        f.write(f"p-value (two-sided, fixed): {p_val_two:.4f}\n")
        f.write(f"p-value (one-sided 'less', scientific): {p_val_one:.4e}\n")
        f.write(f"p-value (one-sided 'less', fixed): {p_val_one:.4f}\n")
    print(f"  -> t-test results saved to {res_path}")

def main():
    DATASETS = ['DIS', 'INH']
    TARGET_COLS = ['PI', 'Ssel', 'WS2', 'RS2']
    FOLD_NUMS = [1, 2, 3, 4, 5]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_STATE = 2025

    for fold_num in FOLD_NUMS:
        for dataset in DATASETS:
            for target_col in TARGET_COLS:
                print(f"\nProcessing Fold: {fold_num} | Dataset: {dataset} | Target: {target_col}")
                
                base_results_dir = f'results/{dataset}/{target_col}/v1/CUE'
                fold_results_dir = os.path.join(base_results_dir, f'fold{fold_num}')
                config_path = os.path.join(base_results_dir, 'config.json')
                
                if not os.path.exists(config_path): continue
                with open(config_path, 'r') as f: train_cfg = json.load(f)

                print("Loading TRAIN analysis data...")
                df = load_train_analysis_data(fold_results_dir, dataset, fold_num)
                df['u_ratio_v2'] = np.log2(df['u_fp'] / df['u_img'].clip(lower=1e-9))

                # Load model and extract embeddings on the fly (matching reference script)
                cfg = CUEConfigs(fold_dir=fold_results_dir, device=DEVICE, hidden_dim=train_cfg.get('hidden_dim', 32), task_type=train_cfg.get('task_type', 'regression'))
                model = CUE(cfg); model.load()

                print("Extracting Train Embeddings on the fly...")
                embs = model.predict(df['SMILES'].values, weights=df[['w_fp', 'w_img']].values)

                save_dir = f"results/analysis/uncertainty/{dataset}/{target_col}/fold{fold_num}/"
                os.makedirs(save_dir, exist_ok=True)

                print("Calculating UMAP (Fit-Transform on Train)...")
                df_umap, axis_range_umap = doUMAP(embs, df[target_col].values, target_col, RANDOM_STATE)
                df['UMAP1'], df['UMAP2'] = df_umap['UMAP1'], df_umap['UMAP2']

                # Plots
                plot_uncertainty_umap(df, save_dir, 'u_ratio_v2', 'Ratio [log2(u_fp/u_img)]', axis_range_umap)
                plot_ratio_vs_selectivity(df, save_dir, target_col)
                plot_prediction_kde(df, save_dir, target_col)
                
                print(f"Completed processing for {dataset} {target_col} fold{fold_num}!")

if __name__ == "__main__":
    main()