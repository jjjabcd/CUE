#!/bin/bash

VERSION="v1"

# 1. Outer loop: Target Columns
TARGET_COLS=(
    "PI" "Ssel"
)

# 2. Inner loop: Datasets
DATA_NAMES=("DIS" "INH")

# The shared ChEMBL tables under chembl/data/selectivity_df keep the original
# assay directory names; every path this project writes uses the unified names.
declare -A DATA_SRC=( [DIS]="Kd" [INH]="Ki" )

# 3. LTAU grid: all (LTAU_BINS, LTAU_K) combinations
LTAU_BINS_LIST=(5 50 50000)
LTAU_K_LIST=(3 5 7)

# LABEL_MODE="zscore"
LABEL_MODE="none"
DEVICE="cuda:0"
SEED=2025

# CUE / Fusion settings
FUSION_POLICY="softmax"
TEMPERATURE=1.0
UNCERTAINTY_NORM="none"

# MLP settings (must match the existing CUE source run being reused)
HIDDEN_DIM=32
NUM_EPOCHS=1000
LEARNING_RATE=1e-4
BATCH_SIZE=256
SAVE_EVERY=1

# Downstream MLP regressor settings
REG_HIDDEN_DIM_1=16
REG_HIDDEN_DIM_2=16
REG_HIDDEN_DIM_3=8
REG_NUM_EPOCHS=500
REG_LEARNING_RATE=1e-3
REG_BATCH_SIZE=256
REG_SAVE_EVERY=1

# --- Start loops ---
for TARGET_COL in "${TARGET_COLS[@]}"; do

    # Determine task type: classification if _label is present
    if [[ "$TARGET_COL" == *"_label"* ]]; then
        TASK_TYPE="classification"
    else
        TASK_TYPE="regression"
    fi

    for DATA_NAME in "${DATA_NAMES[@]}"; do

        DATA_DIR="./../chembl/data/selectivity_df/${DATA_SRC[$DATA_NAME]}"

        # Existing base CUE run: modality-MLP checkpoints + trajectories are reused
        # as-is instead of re-training from scratch for every LTAU grid point.
        CUE_SOURCE_DIR="./results/${DATA_NAME}/${TARGET_COL}/${VERSION}/CUE"

        for LTAU_BINS in "${LTAU_BINS_LIST[@]}"; do
            for LTAU_K in "${LTAU_K_LIST[@]}"; do

                # Directory settings
                OUTPUT_DIR="./results/${DATA_NAME}/${TARGET_COL}/${VERSION}/LTAU_BIN${LTAU_BINS}_K${LTAU_K}"
                LOG_DIR="./logs/${DATA_NAME}/${TARGET_COL}/${VERSION}/LTAU_BIN${LTAU_BINS}_K${LTAU_K}"

                mkdir -p "${LOG_DIR}"
                mkdir -p "${OUTPUT_DIR}"

                LOG_FILE="${LOG_DIR}/${TARGET_COL}.log"

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, TYPE=${TASK_TYPE}, LTAU_BINS=${LTAU_BINS}, LTAU_K=${LTAU_K}" >> "${LOG_FILE}"

                python train_ltau_only.py \
                  --data_dir "${DATA_DIR}" \
                  --output_dir "${OUTPUT_DIR}" \
                  --cue_source_dir "${CUE_SOURCE_DIR}" \
                  --target_col "${TARGET_COL}" \
                  --task_type "${TASK_TYPE}" \
                  --label_mode "${LABEL_MODE}" \
                  --device "${DEVICE}" \
                  --seed "${SEED}" \
                  \
                  --fusion_policy "${FUSION_POLICY}" \
                  --temperature "${TEMPERATURE}" \
                  --uncertainty_norm "${UNCERTAINTY_NORM}" \
                  \
                  --ltau_bins "${LTAU_BINS}" \
                  --ltau_k "${LTAU_K}" \
                  \
                  --hidden_dim "${HIDDEN_DIM}" \
                  --num_epochs "${NUM_EPOCHS}" \
                  --learning_rate "${LEARNING_RATE}" \
                  --batch_size "${BATCH_SIZE}" \
                  --save_every "${SAVE_EVERY}" \
                  \
                  --reg_hidden_dim_1 "${REG_HIDDEN_DIM_1}" \
                  --reg_hidden_dim_2 "${REG_HIDDEN_DIM_2}" \
                  --reg_hidden_dim_3 "${REG_HIDDEN_DIM_3}" \
                  --reg_num_epochs "${REG_NUM_EPOCHS}" \
                  --reg_learning_rate "${REG_LEARNING_RATE}" \
                  --reg_batch_size "${REG_BATCH_SIZE}" \
                  --reg_save_every "${REG_SAVE_EVERY}" \
                  2>&1 | tee -a "${LOG_FILE}"

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, LTAU_BINS=${LTAU_BINS}, LTAU_K=${LTAU_K}" >> "${LOG_FILE}"
                echo -e "\n\n" >> "${LOG_FILE}"

            done
        done
    done
done
