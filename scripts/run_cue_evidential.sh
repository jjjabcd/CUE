#!/bin/bash

VERSION="v1"

# 1. Outer loop: Target Columns
TARGET_COLS=(
    "PI" "Ssel" "WS2" "RS2"
)

# 2. Inner loop: Datasets
DATA_NAMES=("DIS" "INH")

# The shared ChEMBL tables under chembl/data/selectivity_df keep the original
# assay directory names; every path this project writes uses the unified names
# (DIS/INH, WS2/RS2), matching Thesis/CUE/results/<dataset>/<target>/v1/CUE/.
declare -A DATA_SRC=( [DIS]="Kd" [INH]="Ki" )

# LABEL_MODE="zscore"
LABEL_MODE="none"
DEVICE="cuda:0"
SEED=2025

# CUE / Fusion settings
FUSION_POLICY="softmax"
TEMPERATURE=1.0
UNCERTAINTY_NORM="none"

# Uncertainty quantification method: this script always trains with Evidential Regression.
# Uncertainty comes from the predicted NIG parameters in a single forward pass -- no
# dropout sampling and no ensemble, so MC_SAMPLES/ENSEMBLE_SIZE have no role here.
UNCERTAINTY_METHOD="evidential"
METHOD_DIR="Evidential"
# u(x) = beta / (nu * (alpha - 1)), the epistemic term. Deep Ensemble and MC Dropout both
# weight fusion on an epistemic quantity, so matching it keeps the three comparable.
EVIDENTIAL_UNCERTAINTY="epistemic"
EVIDENTIAL_LAMBDA=0.01

# MLP settings
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

    # Evidential Regression is defined for regression targets only; the NIG
    # likelihood has no classification counterpart.
    if [[ "$TARGET_COL" == *"_label"* ]]; then
        echo "[SKIP] ${TARGET_COL}: Evidential Regression does not support classification targets"
        continue
    fi
    TASK_TYPE="regression"


    for DATA_NAME in "${DATA_NAMES[@]}"; do

        # Directory settings: read under the on-disk assay name, write under
        # the unified name (e.g. Kd -> DIS).
        DATA_DIR="./../chembl/data/selectivity_df/${DATA_SRC[$DATA_NAME]}"
        OUTPUT_DIR="./results/${METHOD_DIR}/${DATA_NAME}/${TARGET_COL}/${VERSION}/CUE"
        LOG_DIR="./logs/${METHOD_DIR}/${DATA_NAME}/${TARGET_COL}/${VERSION}"

        mkdir -p "${LOG_DIR}"
        mkdir -p "${OUTPUT_DIR}"

        # Log file name settings: logs/{DATASET}/{TARGET}.log
        LOG_FILE="${LOG_DIR}/${TARGET_COL}.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, TYPE=${TASK_TYPE}" >> "${LOG_FILE}"

        python train.py \
          --data_dir "${DATA_DIR}" \
          --output_dir "${OUTPUT_DIR}" \
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
          --uncertainty_method "${UNCERTAINTY_METHOD}" \
          --evidential_uncertainty "${EVIDENTIAL_UNCERTAINTY}" \
          --evidential_lambda "${EVIDENTIAL_LAMBDA}" \
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

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: DATA=${DATA_NAME}, TARGET=${TARGET_COL}" >> "${LOG_FILE}"
        echo -e "\n\n" >> "${LOG_FILE}"

    done
done
