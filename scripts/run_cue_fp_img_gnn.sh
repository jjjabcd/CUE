#!/bin/bash

VERSION="v1"

# 1. Outer loop: Target Columns
TARGET_COLS=(
    "PI" "Ssel" "WS2" "RS2"
)

# 2. Inner loop: Datasets
DATA_NAMES=("DIS" "INH")

# The shared ChEMBL tables under chembl/data/selectivity_df keep the original
# assay directory names; every path this project writes uses the unified names.
declare -A DATA_SRC=( [DIS]="Kd" [INH]="Ki" )

# Modality combination for this script
MODS="fp img gnn"

LABEL_MODE="none"
DEVICE="cuda:0"
SEED=2025

# CUE / Fusion settings
FUSION_POLICY="softmax"
TEMPERATURE=1.0
UNCERTAINTY_NORM="none"

# Uncertainty method: ltau | mc_dropout | deep_ensemble (freely combinable with MODS above)
UNCERTAINTY_METHOD="ltau"

# Which pretrained GNN backs the 'gnn' modality (only relevant if MODS includes gnn): mole | mgssl
GNN_BACKEND="mole"

# LTAU settings (used when UNCERTAINTY_METHOD=ltau)
LTAU_K=5
LTAU_BINS=5000

# MC Dropout settings (used when UNCERTAINTY_METHOD=mc_dropout)
MC_SAMPLES=10
DROPOUT_RATE=0.1

# Deep Ensemble settings (used when UNCERTAINTY_METHOD=deep_ensemble)
ENSEMBLE_SIZE=10
ENSEMBLE_SEEDS="1 2 3 4 5 6 7 8 9 10"

# Modality-MLP pretrain settings
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

    if [[ "$TARGET_COL" == *"_label"* ]]; then
        TASK_TYPE="classification"
    else
        TASK_TYPE="regression"
    fi

    for DATA_NAME in "${DATA_NAMES[@]}"; do

        DATA_DIR="./../chembl/data/selectivity_df/${DATA_SRC[$DATA_NAME]}"

        MOD_NAME="${MODS^^}"
        MOD_NAME="${MOD_NAME// /+}"

        OUTPUT_DIR="./results/${DATA_NAME}/${TARGET_COL}/${VERSION}/${MOD_NAME}"
        LOG_DIR="./logs/${DATA_NAME}/${TARGET_COL}/${VERSION}"

        mkdir -p "${LOG_DIR}"
        mkdir -p "${OUTPUT_DIR}"

        LOG_FILE="${LOG_DIR}/${MOD_NAME}.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, TYPE=${TASK_TYPE}, MODALITY=${MOD_NAME}" >> "${LOG_FILE}"

        python FP_IMG_GNN_train.py \
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
          --gnn_backend "${GNN_BACKEND}" \
          --ltau_bins "${LTAU_BINS}" \
          --ltau_k "${LTAU_K}" \
          --mc_samples "${MC_SAMPLES}" \
          --dropout_rate "${DROPOUT_RATE}" \
          --ensemble_size "${ENSEMBLE_SIZE}" \
          --ensemble_seeds ${ENSEMBLE_SEEDS} \
          \
          --hidden_dim "${HIDDEN_DIM}" \
          --num_epochs "${NUM_EPOCHS}" \
          --learning_rate "${LEARNING_RATE}" \
          --batch_size "${BATCH_SIZE}" \
          --save_every "${SAVE_EVERY}" \
          \
          --active_modalities ${MODS} \
          \
          --reg_hidden_dim_1 "${REG_HIDDEN_DIM_1}" \
          --reg_hidden_dim_2 "${REG_HIDDEN_DIM_2}" \
          --reg_hidden_dim_3 "${REG_HIDDEN_DIM_3}" \
          --reg_num_epochs "${REG_NUM_EPOCHS}" \
          --reg_learning_rate "${REG_LEARNING_RATE}" \
          --reg_batch_size "${REG_BATCH_SIZE}" \
          --reg_save_every "${REG_SAVE_EVERY}" \
          2>&1 | tee -a "${LOG_FILE}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, MODALITY=${MOD_NAME}" >> "${LOG_FILE}"
        echo -e "\n\n" >> "${LOG_FILE}"

    done
done
