"""
Pre-extract FP(Morgan) / IMG(MolScribe) / SMILES(ChemBERTa) / GNN(MolE) /
Conformation(ConforFormer) features for every (dataset, fold) group under
cue4modal's selectivity data directory.

Mirrors udmev7/udme/main/prepare_features.py, adapted to cue4modal's actual
data layout (multi-target CSVs, not one label per directory):

    <data_dir>/
        <Kd|Ki>/
            fold1/ fold2/ ... fold5/
                selectivity_train.csv   (columns: compound_chembl_id, SMILES, <many target cols>)
                selectivity_val.csv
                selectivity_test.csv

For every split CSV a paired .npy is written next to it:
    fold1/selectivity_train.npy  <- np.load(..., allow_pickle=True).item() -> dict with:
        'id'                : list[str]                  (compound_chembl_id)
        'smiles'            : list[str]
        <original target columns, unchanged, one entry per column>
        'fp_features'        : np.ndarray (N, 2048) (Morgan)
        'img_features'       : np.ndarray (N, 1024) (MolScribe)
        'smi_features_mean'  : np.ndarray (N, 768)
        'smi_features_cls'   : np.ndarray (N, 768)
        'gnn_features'       : np.ndarray (N, 768)  (MolE)
        'gnn_features_mgssl' : np.ndarray (N, 300)  (MGSSL, alternate GNN backend)
        'conf_features'      : np.ndarray (N, 512)

Usage:
    python -m cue.utils.prepare_features --data_dir ../chembl/data/selectivity_df --device cuda:0
    python -m cue.utils.prepare_features --dataset Kd --overwrite
    python -m cue.utils.prepare_features --skip_existing_keys --modalities fp img smi gnn
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cue.utils.FeatureExtraction import MolEncoder

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ID_COL = "compound_chembl_id"
SMILES_COL = "SMILES"
SPLITS = ("train", "val", "test")

ALL_MODALITIES = ("fp", "img", "smi", "gnn", "gnn_mgssl", "conf")
MODALITY_KEYS = {
    "fp": {"fp_features"},
    "img": {"img_features"},
    "smi": {"smi_features_mean", "smi_features_cls"},
    "gnn": {"gnn_features"},
    "gnn_mgssl": {"gnn_features_mgssl"},
    "conf": {"conf_features"},
}


def find_fold_dirs(data_dir: Path, filter_dataset: str | None) -> List[Path]:
    fold_dirs: List[Path] = []
    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if filter_dataset and dataset_dir.name != filter_dataset:
            continue
        for fold_dir in sorted(dataset_dir.iterdir()):
            if fold_dir.is_dir() and fold_dir.name.startswith("fold"):
                fold_dirs.append(fold_dir)
    return fold_dirs


def _read_split(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def encode_unique_smiles(
    encoder: MolEncoder,
    smiles_list: List[str],
    modalities: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Encode a deduplicated list of SMILES strings for the requested modalities.

    Returns:
        dict  smiles -> {'fp':(2048,), 'img':(1024,), 'smi_mean':(768,), 'smi_cls':(768,),
                         'gnn':(768,), 'gnn_mgssl':(300,), 'conf':(512,)}
        (only the keys for the requested modalities are populated)
    """
    unique = list(dict.fromkeys(s for s in smiles_list if isinstance(s, str) and s))
    U = len(unique)
    lookup: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in unique}

    if "fp" in modalities:
        for i, smi in enumerate(tqdm(unique, desc="    FP (Morgan)", leave=False)):
            try:
                fp_emb = encoder.get_fpt_features(smi)
            except Exception as e:
                tqdm.write(f"    [WARN] fp '{smi}': {e}")
                fp_emb = np.zeros(encoder.feature_dim_fp, dtype=np.int64)
            lookup[smi]["fp"] = fp_emb

    if "img" in modalities:
        for i, smi in enumerate(tqdm(unique, desc="    IMG (MolScribe)", leave=False)):
            try:
                img_emb = encoder.get_img_features(smi)
            except Exception as e:
                tqdm.write(f"    [WARN] img '{smi}': {e}")
                img_emb = np.zeros(encoder.feature_dim_img, dtype=np.float32)
            lookup[smi]["img"] = img_emb

    if "gnn" in modalities:
        gnn_all = encoder.get_gnn_features_all(unique)  # (U, 768)
        for i, smi in enumerate(unique):
            lookup[smi]["gnn"] = gnn_all[i]

    if "gnn_mgssl" in modalities:
        gnn_mgssl_all = encoder.get_gnn_features_mgssl_all(unique)  # (U, 300)
        for i, smi in enumerate(unique):
            lookup[smi]["gnn_mgssl"] = gnn_mgssl_all[i]

    if "smi" in modalities:
        for i, smi in enumerate(tqdm(unique, desc="    SMI (ChemBERTa)", leave=False)):
            try:
                cls_emb, mean_emb = encoder.get_smi_features_both(smi)
            except Exception as e:
                tqdm.write(f"    [WARN] smi '{smi}': {e}")
                cls_emb = np.zeros(encoder.feature_dim_smi, dtype=np.float32)
                mean_emb = np.zeros(encoder.feature_dim_smi, dtype=np.float32)
            lookup[smi]["smi_cls"] = cls_emb
            lookup[smi]["smi_mean"] = mean_emb

    if "conf" in modalities:
        for i, smi in enumerate(tqdm(unique, desc="    Conformation (ConforFormer)", leave=False)):
            try:
                conf_emb = encoder.get_conf_features(smi)
            except Exception as e:
                tqdm.write(f"    [WARN] conf '{smi}': {e}")
                conf_emb = np.zeros(encoder.feature_dim_conf, dtype=np.float32)
            lookup[smi]["conf"] = conf_emb

    return lookup


def process_fold(
    fold_dir: Path,
    encoder: MolEncoder,
    overwrite: bool,
    skip_existing_keys: bool,
    modalities: List[str],
) -> None:
    # (split, csv_path, npy_path, df, modalities actually missing/needed for THIS split)
    splits_to_process: List[Tuple[str, Path, Path, pd.DataFrame, List[str]]] = []
    all_smiles: List[str] = []
    modalities_needed_overall: set = set()

    for split in SPLITS:
        csv_path = fold_dir / f"selectivity_{split}.csv"
        if not csv_path.exists():
            continue

        npy_path = csv_path.with_suffix(".npy")

        if not npy_path.exists() or overwrite:
            modalities_needed = list(modalities)
        elif skip_existing_keys:
            try:
                existing = np.load(npy_path, allow_pickle=True).item()
                existing_keys = set(existing.keys())
                # Only the modalities whose keys are actually absent need (re)computing -
                # e.g. adding a new modality to an already-cached fold must not force every
                # other already-cached modality to be recomputed too.
                modalities_needed = [m for m in modalities if MODALITY_KEYS[m] - existing_keys]
            except Exception:
                modalities_needed = list(modalities)
        else:
            modalities_needed = []

        if not modalities_needed:
            continue

        df = _read_split(csv_path)
        all_smiles.extend(df[SMILES_COL].astype(str).tolist())
        modalities_needed_overall |= set(modalities_needed)
        splits_to_process.append((split, csv_path, npy_path, df, modalities_needed))

    if not splits_to_process:
        return

    tqdm.write(
        f"  [ENC] {fold_dir.parent.name}/{fold_dir.name}  "
        f"({len(set(all_smiles))} unique SMILES, modalities={sorted(modalities_needed_overall)})"
    )
    lookup = encode_unique_smiles(encoder, all_smiles, list(modalities_needed_overall))

    zero_fp = np.zeros(encoder.feature_dim_fp, dtype=np.int64)
    zero_img = np.zeros(encoder.feature_dim_img, dtype=np.float32)
    zero_smi = np.zeros(encoder.feature_dim_smi, dtype=np.float32)
    zero_gnn = np.zeros(encoder.feature_dim_gnn, dtype=np.float32)
    zero_gnn_mgssl = np.zeros(encoder.feature_dim_gnn_mgssl, dtype=np.float32)
    zero_conf = np.zeros(encoder.feature_dim_conf, dtype=np.float32)

    for split, csv_path, npy_path, df, modalities_needed in splits_to_process:
        smiles = df[SMILES_COL].astype(str).tolist()

        new_data: Dict[str, object] = {
            "id": df[ID_COL].astype(str).tolist(),
            "smiles": smiles,
        }
        # Preserve every other (target) column untouched.
        for col in df.columns:
            if col in (ID_COL, SMILES_COL):
                continue
            new_data[col] = df[col].tolist()

        if "fp" in modalities_needed:
            new_data["fp_features"] = np.stack(
                [lookup.get(s, {}).get("fp", zero_fp) for s in smiles]
            )
        if "img" in modalities_needed:
            new_data["img_features"] = np.stack(
                [lookup.get(s, {}).get("img", zero_img) for s in smiles]
            )
        if "smi" in modalities_needed:
            new_data["smi_features_mean"] = np.stack(
                [lookup.get(s, {}).get("smi_mean", zero_smi) for s in smiles]
            )
            new_data["smi_features_cls"] = np.stack(
                [lookup.get(s, {}).get("smi_cls", zero_smi) for s in smiles]
            )
        if "gnn" in modalities_needed:
            new_data["gnn_features"] = np.stack(
                [lookup.get(s, {}).get("gnn", zero_gnn) for s in smiles]
            )
        if "gnn_mgssl" in modalities_needed:
            new_data["gnn_features_mgssl"] = np.stack(
                [lookup.get(s, {}).get("gnn_mgssl", zero_gnn_mgssl) for s in smiles]
            )
        if "conf" in modalities_needed:
            new_data["conf_features"] = np.stack(
                [lookup.get(s, {}).get("conf", zero_conf) for s in smiles]
            )

        if npy_path.exists() and not overwrite and skip_existing_keys:
            try:
                existing = np.load(npy_path, allow_pickle=True).item()
                existing.update(new_data)
                new_data = existing
            except Exception as e:
                tqdm.write(f"    [ERR] Could not merge with existing .npy at {npy_path}: {e}")

        np.save(npy_path, new_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-extract SMILES/GNN/Conformation features for cue4modal selectivity folds."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "chembl" / "data" / "selectivity_df",
        help="Root directory containing <Kd|Ki>/fold*/selectivity_{train,val,test}.csv",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Re-compute even if .npy files already exist.")
    parser.add_argument("--skip_existing_keys", action="store_true", help="If .npy exists, only compute missing feature keys.")
    parser.add_argument("--dataset", type=str, default=None, help="Process only this dataset (e.g. 'Kd').")
    parser.add_argument(
        "--modalities", type=str, nargs="+", default=list(ALL_MODALITIES), choices=list(ALL_MODALITIES),
        help="Which modalities to extract/cache.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"[ERROR] data_dir not found: {args.data_dir}")

    import torch
    device = torch.device(args.device)

    print(f"Loading MolEncoder on {device} ...")
    encoder = MolEncoder(device=device)
    print("MolEncoder ready.\n")

    fold_dirs = find_fold_dirs(args.data_dir, args.dataset)
    if not fold_dirs:
        raise SystemExit("[ERROR] No matching dataset/fold directories found.")

    print(f"Found {len(fold_dirs)} fold directorie(s). Modalities: {args.modalities}\n")

    for fold_dir in tqdm(fold_dirs, desc="Folds"):
        try:
            process_fold(
                fold_dir=fold_dir,
                encoder=encoder,
                overwrite=args.overwrite,
                skip_existing_keys=args.skip_existing_keys,
                modalities=args.modalities,
            )
        except Exception as e:
            tqdm.write(f"  [ERROR] {fold_dir}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
