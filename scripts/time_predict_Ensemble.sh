#!/bin/bash

# ==============================================================================
# Prediction speed benchmark: cue4unc Deep_Ensemble
# ==============================================================================
# Run from the cue4unc/ project root: bash scripts/time_predict_Ensemble.sh
# Saves per-10k-molecule timing checkpoints to cue/results/time/timestep_deep_ensemble.csv.
# Also runs the uncertainty sanity check (10 distinct ensemble members) before
# timing starts.
# ==============================================================================

DATASET="DIS"
TARGET="PI"
FOLD="fold1"
VERSION="v1"
SMILES_COL="SMILES"
DEVICE="cuda:0"
BATCH_SIZE=128
CHECKPOINT_INTERVAL=10000

INPUT_FILE="../pubchem/train_1m.csv"
OUTPUT_CSV="../cue/results/time/timestep_deep_ensemble.csv"
mkdir -p "$(dirname "${OUTPUT_CSV}")"

python time_predict.py \
    --base_dir "./results/Deep_Ensemble" \
    --data_name "${DATASET}" \
    --target_cols "${TARGET}" \
    --version "${VERSION}" \
    --specific_fold "${FOLD}" \
    --input_file "${INPUT_FILE}" \
    --smiles_col "${SMILES_COL}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --output_csv "${OUTPUT_CSV}"
