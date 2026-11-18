#!/bin/bash

TARGET='6S89'
MODEL='CUE'
TRIAL='Trial_2'
SCORE_COLUMN='Proxy'

INPUT_FILE="/data/rlawlsgurjh/work/gitlab/CUE/cue_vs/results/${TARGET}/${MODEL}/${TRIAL}/${SCORE_COLUMN}/03_dta_filtered.csv"
SMILES_COL="SMILES"

# Target Columns
# PI만 먼저 분석 진행
TARGET_COLS=(
    "PI" "Ssel" "WS2" "RS2"
)

# Datasets (Models are trained on these)
DATA_NAMES=("DIS")

# Versioning
VERSION="v1"

# CUE Name (Assuming using the base CUE model)
MODEL_NAME="CUE"

# Output Directory for predictions
OUTPUT_BASE="./results/virtual_screening/${MODEL}/${TARGET}/${TRIAL}"
mkdir -p "${OUTPUT_BASE}"

DEVICE="cuda:0"

BATCH_SIZE="1024"

# Loop
for DATA_NAME in "${DATA_NAMES[@]}"; do
    for TARGET_COL in "${TARGET_COLS[@]}"; do
        
        echo "----------------------------------------------------------------"
        echo "Processing ${DATA_NAME} | ${TARGET_COL} | ${VERSION}"
        echo "----------------------------------------------------------------"

        python predict.py \
            --input_file "${INPUT_FILE}" \
            --smiles_col "${SMILES_COL}" \
            --base_dir "./results" \
            --data_name "${DATA_NAME}" \
            --target_col "${TARGET_COL}" \
            --version "${VERSION}" \
            --model_name "${MODEL_NAME}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --specific_fold "fold1"  # Uncomment to use only fold1

        # Move correct output to organized folder
        # predict.py outputs to current directory: prediction_{DATA_NAME}_{TARGET_COL}_{VERSION}_{ABLATION_NAME}.csv
        
        PRED_FILE="prediction_${DATA_NAME}_${TARGET_COL}_${VERSION}_${MODEL_NAME}.csv"
        if [ -f "${PRED_FILE}" ]; then
            mv "${PRED_FILE}" "${OUTPUT_BASE}/${PRED_FILE}"
            echo "Saved to ${OUTPUT_BASE}/${PRED_FILE}"
        else
            echo "[WARN] Prediction file not found: ${PRED_FILE}"
        fi

    done
done
