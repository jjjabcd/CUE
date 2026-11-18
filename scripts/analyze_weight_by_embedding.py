import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles

from sklearn.decomposition import PCA

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
from cue.utils.FeatureExtraction import MolEncoder


def _sanitize_embs(embs):
    embs = np.asarray(embs, dtype=np.float32)
    if np.any(np.isnan(embs)) or np.any(np.isinf(embs)):
        embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    return embs

def doUMAP(tr_embs, tr_y, target_col, random_state):
    tr_embs = _sanitize_embs(tr_embs)

    n_neighbors = min(15, max(2, len(tr_embs) - 1))

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.1
    )

    reducer.fit(tr_embs)
    tr_points = reducer.transform(tr_embs)

    df_tr = pd.DataFrame({"UMAP1": tr_points[:, 0], "UMAP2": tr_points[:, 1], target_col: tr_y})

    x_min, x_max = tr_points[:, 0].min(), tr_points[:, 0].max()
    y_min, y_max = tr_points[:, 1].min(), tr_points[:, 1].max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    axis_range = {
        "x_min": x_min - x_pad,
        "x_max": x_max + x_pad,
        "y_min": y_min - y_pad,
        "y_max": y_max + y_pad,
    }

    return df_tr, axis_range

def doPCA(tr_embs, tr_y, target_col, random_state):
    tr_embs = _sanitize_embs(tr_embs)
    
    pca = PCA(n_components=2, random_state=random_state)
    tr_points = pca.fit_transform(tr_embs)

    df_tr = pd.DataFrame({
        "PC1": tr_points[:, 0],
        "PC2": tr_points[:, 1],
        target_col: tr_y
    })

    x_min, x_max = tr_points[:, 0].min(), tr_points[:, 0].max()
    y_min, y_max = tr_points[:, 1].min(), tr_points[:, 1].max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    axis_range = {
        "x_min": x_min - x_pad,
        "x_max": x_max + x_pad,
        "y_min": y_min - y_pad,
        "y_max": y_max + y_pad,
    }

    return df_tr, axis_range

def scatterplot(df, filepath, target_col, target_type, axis_range, title='Embedding Distribution', x_col='UMAP1', y_col='UMAP2'):
    if target_type == 'regression':
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Regression: 기존처럼 rainbow colormap 사용
        scatter = ax.scatter(df[x_col], df[y_col], c=df[target_col], cmap='rainbow', alpha=0.8, s=20)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(target_col, fontsize=12)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Classification: 0=Blue, 1=Red 로 이진 분류 시각화
        colors = {0: 'blue', 1: 'red'}
        
        # 각 클래스별로 scatter plot을 그려서 legend 생성
        unique_labels = sorted(df[target_col].unique())
        for label in unique_labels:
            subset = df[df[target_col] == label]
            color_val = colors.get(int(label), 'gray') # 0, 1 이외의 값이면 회색
            ax.scatter(subset[x_col], subset[y_col], c=color_val, label=f'Class {int(label)}', alpha=0.8, s=20)
            
        # ax.legend() # 범례 추가

    ax.set_xlim(axis_range['x_min'], axis_range['x_max'])
    ax.set_ylim(axis_range['y_min'], axis_range['y_max'])

    # 공통 축 라벨 및 타이틀 설정
    ax.set_xlabel(x_col, fontsize=15)
    ax.set_ylabel(y_col, fontsize=15)
    # ax.set_title(title, fontsize=16)
    
    # plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def load_data(dataset, target_col, fold_num):
    # the shared ChEMBL tables keep the on-disk assay folder names (DIS -> Kd)
    data_dir = f'../chembl/data/selectivity_df/{DATASET_TO_SOURCE.get(dataset, dataset)}/fold{fold_num}'
    df_tr = normalize_columns(pd.read_csv(os.path.join(data_dir, 'selectivity_train.csv')))
    df_tr['original_order'] = range(len(df_tr))

    fusion_dir = f'results/{dataset}/{target_col}/v1/CUE/fold{fold_num}/fusion'
    f_tr = pd.read_csv(os.path.join(fusion_dir, 'train_fusion.csv'))
    
    # Merge on compound_chembl_id to get w_fp and w_img
    df_tr = pd.merge(df_tr, f_tr[['compound_chembl_id', 'w_fp', 'w_img']], on='compound_chembl_id', how='left')
    df_tr = df_tr.sort_values('original_order').drop(columns=['original_order']).reset_index(drop=True)
    
    return df_tr

def process_and_plot(tr_embs, df_tr, save_dir, target_col, color_col='w_fp', random_state=2025):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"  -> Running PCA (color={color_col})...")
    # PCA
    df_tr_pca, axis_range_pca = doPCA(
        tr_embs, 
        df_tr[color_col].values, 
        target_col=color_col, random_state=random_state
    )
    scatterplot(df_tr_pca, os.path.join(save_dir, f'{target_col}_train_pca_{color_col}.png'), color_col, 'regression', axis_range_pca, f'Train PCA ({target_col})', 'PC1', 'PC2')

    print(f"  -> Running UMAP (color={color_col})...")
    # UMAP
    df_tr_umap, axis_range_umap = doUMAP(
        tr_embs, 
        df_tr[color_col].values, 
        target_col=color_col, random_state=random_state
    )
    scatterplot(df_tr_umap, os.path.join(save_dir, f'{target_col}_train_umap_{color_col}.png'), color_col, 'regression', axis_range_umap, f'Train UMAP ({target_col})', 'UMAP1', 'UMAP2')
    print("  -> Done plotting.")


def analyze_fp(dataset, target_col, fold_num, df_tr):
    '''
    Method 1: Raw Fingerprint
    '''
    print(f"[{dataset}/{target_col}] Method 1: Raw Fingerprint")
    encoder = MolEncoder()
    tr_embs = np.array([encoder.get_fpt_features(s) for s in df_tr['SMILES']])
    
    save_dir = f"results/weight/emb/{dataset}/fold{fold_num}/1/"
    process_and_plot(tr_embs, df_tr, save_dir, target_col)

def analyze_img(dataset, target_col, fold_num, df_tr):
    '''
    Method 2: Raw Image Feature
    '''
    print(f"[{dataset}/{target_col}] Method 2: Raw Image Feature")
    encoder = MolEncoder()
    tr_embs = np.array([encoder.get_img_features(s) for s in df_tr['SMILES']])
    
    save_dir = f"results/weight/emb/{dataset}/fold{fold_num}/2/"
    process_and_plot(tr_embs, df_tr, save_dir, target_col, color_col='w_fp')
    process_and_plot(tr_embs, df_tr, save_dir, target_col, color_col='w_img')

def analyze_fp_emb(dataset, target_col, fold_num, df_tr, model):
    '''
    Method 3: Fingerprint Embedding Vector
    '''
    print(f"[{dataset}/{target_col}] Method 3: FP Embedding")
    tr_embs = model.get_modality_descriptors(df_tr['SMILES'].values)['fp']
    
    save_dir = f"results/weight/emb/{dataset}/fold{fold_num}/3/"
    process_and_plot(tr_embs, df_tr, save_dir, target_col)

def analyze_img_emb(dataset, target_col, fold_num, df_tr, model):
    '''
    Method 4: Image Embedding Vector
    '''
    print(f"[{dataset}/{target_col}] Method 4: IMG Embedding")
    tr_embs = model.get_modality_descriptors(df_tr['SMILES'].values)['img']
    
    save_dir = f"results/weight/emb/{dataset}/fold{fold_num}/4/"
    process_and_plot(tr_embs, df_tr, save_dir, target_col, color_col='w_fp')
    process_and_plot(tr_embs, df_tr, save_dir, target_col, color_col='w_img')

def analyze_unified_emb(dataset, target_col, fold_num, df_tr, model):
    '''
    Method 5: Unified Embedding Vector
    '''
    print(f"[{dataset}/{target_col}] Method 5: Unified Embedding")
    # Use weighted average unified embedding to be consistent with analyze_unc.py
    weights_tr = df_tr[['w_fp', 'w_img']].values
    tr_embs = model.predict(df_tr['SMILES'].values, weights=weights_tr)

    save_dir = f"results/weight/emb/{dataset}/fold{fold_num}/5/"
    process_and_plot(tr_embs, df_tr, save_dir, target_col)


def main():
    datasets = ['DIS', 'INH']
    target_cols = ['PI', 'Ssel', 'WS2', 'RS2']
    fold_nums = [1, 2, 3, 4, 5] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold_num in fold_nums:
        for dataset in datasets:
            for target_col in target_cols:
                print(f"\n{'='*50}\nProcessing Fold: {fold_num} | Dataset: {dataset} | Target: {target_col}\n{'='*50}")
                print("Loading data...")
                df_tr = load_data(dataset, target_col, fold_num)
                
                # Load model for embedding extraction
                results_dir = f'results/{dataset}/{target_col}/v1/CUE/fold{fold_num}'
                cfg = CUEConfigs(fold_dir=results_dir, device=device)
                model = CUE(cfg)
                print("Loading CUE Model checkpoints...")
                model.load()

                # Execute all Methods
                analyze_fp(dataset, target_col, fold_num, df_tr)
                analyze_img(dataset, target_col, fold_num, df_tr)
                analyze_fp_emb(dataset, target_col, fold_num, df_tr, model)
                analyze_img_emb(dataset, target_col, fold_num, df_tr, model)
                analyze_unified_emb(dataset, target_col, fold_num, df_tr, model)

if __name__ == "__main__":
    main()