#!/bin/bash

VERSION="v1"

# 1. Outer loop: Target Columns
TARGET_COLS=(
    "WS2" "RS2"
)

# 2. Inner loop: Datasets
DATA_NAMES=("DIS" "INH")

# The shared ChEMBL tables under chembl/data/selectivity_df keep the original
# assay directory names; every path this project writes uses the unified names.
declare -A DATA_SRC=( [DIS]="Kd" [INH]="Ki" )

# LABEL_MODE="zscore"
LABEL_MODE="none"
DEVICE="cuda:1"

# CUE / Fusion settings
# fusion_policy="random": modality 가중치를 샘플별로 계산하지 않고, 1회 실행당
# Dirichlet(1,...,1)에서 한 번만 뽑아 모든 샘플에 동일하게 고정 적용한다 (cue/core/fusion.py 참고).
FUSION_POLICY="random"
TEMPERATURE=1.0
UNCERTAINTY_NORM="none"

# LTAU settings
LTAU_K=5
LTAU_BINS=5000

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

    # Determine task type: classification if _label is present
    if [[ "$TARGET_COL" == *"_label"* ]]; then
        TASK_TYPE="classification"
    else
        TASK_TYPE="regression"
    fi

    for DATA_NAME in "${DATA_NAMES[@]}"; do

        DATA_DIR="./../chembl/data/selectivity_df/${DATA_SRC[$DATA_NAME]}"

        # random 가중치를 10번(SEED=11..30) 뽑아 각각 R11..R30 폴더에 저장한다.
        for i in $(seq 11 30); do
            SEED=$i

            # Directory settings
            OUTPUT_DIR="./results/${DATA_NAME}/${TARGET_COL}/${VERSION}/R${i}"
            LOG_DIR="./logs/${DATA_NAME}/${TARGET_COL}/${VERSION}/R${i}"

            mkdir -p "${LOG_DIR}"
            mkdir -p "${OUTPUT_DIR}"

            LOG_FILE="${LOG_DIR}/${TARGET_COL}.log"

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, TYPE=${TASK_TYPE}, SEED=${SEED}" >> "${LOG_FILE}"

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

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: DATA=${DATA_NAME}, TARGET=${TARGET_COL}, SEED=${SEED}" >> "${LOG_FILE}"
            echo -e "\n\n" >> "${LOG_FILE}"

        done
    done
done
