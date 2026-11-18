#!/bin/bash

INPUT_FILE="./../chembl/data/selectivity_df/davis_selectivity.csv"
SMILES_COL="SMILES"

# Target Columns
TARGET_COLS=(
    "PI" "Ssel" "WS2" "RS2"
)

# Dataset (models are trained on this)
DATA_NAME="DIS"

# Versioning
VERSION="v1"

# CUE Name (Assuming using the base CUE model)
MODEL_NAME="CUE"

DEVICE="cuda:0"
BATCH_SIZE="1024"

FOLDS=("fold1" "fold2" "fold3" "fold4" "fold5")

# Output directory for davis fold-level predictions
OUTPUT_BASE="./results/davis_pred/${DATA_NAME}"
mkdir -p "${OUTPUT_BASE}"

for FOLD in "${FOLDS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Processing ${DATA_NAME} | ${FOLD}"
    echo "----------------------------------------------------------------"

    python predict.py \
        --input_file "${INPUT_FILE}" \
        --smiles_col "${SMILES_COL}" \
        --base_dir "./results" \
        --data_name "${DATA_NAME}" \
        --target_cols "${TARGET_COLS[@]}" \
        --version "${VERSION}" \
        --model_name "${MODEL_NAME}" \
        --device "${DEVICE}" \
        --batch_size "${BATCH_SIZE}" \
        --specific_fold "${FOLD}"

    PRED_FILE="prediction_${DATA_NAME}_${MODEL_NAME}.csv"
    FOLD_OUT_DIR="${OUTPUT_BASE}/${FOLD}"
    mkdir -p "${FOLD_OUT_DIR}"
    if [ -f "${PRED_FILE}" ]; then
        mv "${PRED_FILE}" "${FOLD_OUT_DIR}/prediction.csv"
        echo "Saved to ${FOLD_OUT_DIR}/prediction.csv"
    else
        echo "[WARN] Prediction file not found: ${PRED_FILE}"
    fi
done
