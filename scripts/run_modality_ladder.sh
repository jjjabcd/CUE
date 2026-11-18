#!/bin/bash
# Modality ladder: start from the two best unimodal encoders and add one modality at a time,
# crossed with the uncertainty quantification methods.
#
# Unimodal ranking over the 8 CUE datasets (paper/notebook/modal.ipynb, average rank over
# RMSE / R2 / PCC):  FP 1.2  >  GNN(MolE) 2.3  >  IMG 3.5  >  Conf 3.7  >  SMI 4.3
# so the ladder adds modalities in that order and FP+GNN is the default.
#
# Output: ./results/<UQ>/<DIS|INH>/<target>/v1/<FP+GNN|...>
#   (new tree; the existing ./results/<DIS|INH>/... LTAU runs are left untouched)
#
# Usage:
#   bash scripts/run_modality_ladder.sh
#   COMBOS=0 UQ_METHODS="mc_dropout" bash scripts/run_modality_ladder.sh   # one rung, one UQ
#   DRY_RUN=1 bash scripts/run_modality_ladder.sh                          # print only

set -uo pipefail

cd "$(dirname "$0")/.."

VERSION="v1"
# Explicit interpreter: the script is usually launched without `conda activate`, and a bare
# `python` then resolves to the base environment, which has no numpy/torch.
PYTHON="${PYTHON:-$HOME/miniconda3/envs/cue/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED=2025
# Seed for the downstream head's initialization, applied identically to every cell. Kept separate
# from SEED so the head's init does not depend on how much of the RNG stream the upstream stages
# consumed, which previously varied with the modality count and drove whole cells into a
# dead-ReLU constant-output collapse. Never tune it per cell.
HEAD_SEED="${HEAD_SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
# Re-run cells that already have metrics.csv (default: skip them).
FORCE="${FORCE:-0}"

# The ladder, cumulative. Override with COMBOS="0 1" to run only some rungs.
ALL_COMBOS=(
    "fp gnn"
    "fp gnn img"
    "fp gnn img conf"
    "fp gnn img conf smi"
)
COMBOS="${COMBOS:-0 1 2 3}"

# Deep Ensemble trains `ensemble_size` members per modality and measures ~5x the others
# (cue4unc logs: 52 min/run vs 10-11), so it runs last and the cheap methods land first.
UQ_METHODS="${UQ_METHODS:-ltau mc_dropout evidential deep_ensemble}"

TARGET_COLS=("PI" "Ssel" "WS2" "RS2")
DATA_NAMES=("DIS" "INH")

# The input folders keep the on-disk assay names; everything written uses the unified ones.
declare -A DATA_SRC=( [DIS]="Kd" [INH]="Ki" )
uq_outname() {
    case "$1" in
        ltau)          echo "LTAU" ;;
        mc_dropout)    echo "MC_Dropout" ;;
        deep_ensemble) echo "Deep_Ensemble" ;;
        evidential)    echo "Evidential" ;;
        *)             echo "$1" ;;
    esac
}

LABEL_MODE="none"
FUSION_POLICY="softmax"
TEMPERATURE=1.0
UNCERTAINTY_NORM="none"

# The 'gnn' modality is MolE, which ranked second among the unimodal encoders (MGSSL was worse).
GNN_BACKEND="mole"

LTAU_K=5
LTAU_BINS=5000
MC_SAMPLES=10
# MC Dropout only: model.py uses cfg.dropout_rate solely when uncertainty_method == 'mc_dropout'
# and hardwires 0.0 for LTAU / Deep Ensemble / Evidential, so this value never reaches the other
# three methods. 0.2 matches the main comparison table (cue4unc MC_Dropout v1), so the ladder's
# MC Dropout rungs start from the same model that table reports.
DROPOUT_RATE="${DROPOUT_RATE:-0.2}"
ENSEMBLE_SIZE=10
ENSEMBLE_SEEDS="1 2 3 4 5 6 7 8 9 10"
EVIDENTIAL_LAMBDA=0.01
EVIDENTIAL_UNCERTAINTY="epistemic"

HIDDEN_DIM=32
NUM_EPOCHS=1000
LEARNING_RATE=1e-4
BATCH_SIZE=256
SAVE_EVERY=1

REG_HIDDEN_DIM_1=16
REG_HIDDEN_DIM_2=16
REG_HIDDEN_DIM_3=8
REG_NUM_EPOCHS=500
REG_LEARNING_RATE=1e-3
REG_BATCH_SIZE=256
REG_SAVE_EVERY=1

failures=0

for UQ in ${UQ_METHODS}; do
    UQ_DIR="$(uq_outname "${UQ}")"

    for idx in ${COMBOS}; do
        MODS="${ALL_COMBOS[$idx]}"
        MOD_NAME="${MODS^^}"
        MOD_NAME="${MOD_NAME// /+}"

        for TARGET_COL in "${TARGET_COLS[@]}"; do
            TASK_TYPE="regression"
            [[ "$TARGET_COL" == *"_label"* ]] && TASK_TYPE="classification"

            for DATA_NAME in "${DATA_NAMES[@]}"; do
                DATA_DIR="./../chembl/data/selectivity_df/${DATA_SRC[$DATA_NAME]}"
                OUTPUT_DIR="./results/${UQ_DIR}/${DATA_NAME}/${TARGET_COL}/${VERSION}/${MOD_NAME}"
                LOG_DIR="./logs/${UQ_DIR}/${DATA_NAME}/${TARGET_COL}/${VERSION}"
                LOG_FILE="${LOG_DIR}/${MOD_NAME}.log"

                if [[ -f "${OUTPUT_DIR}/metrics.csv" && "${FORCE}" != "1" ]]; then
                    echo "[SKIP] ${UQ_DIR} ${MOD_NAME} ${DATA_NAME}/${TARGET_COL}: metrics.csv exists"
                    continue
                fi

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${UQ_DIR} | ${MOD_NAME} | ${DATA_NAME}/${TARGET_COL}"
                if [[ "${DRY_RUN}" == "1" ]]; then
                    continue
                fi

                mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${UQ} ${MOD_NAME} ${DATA_NAME}/${TARGET_COL}" >> "${LOG_FILE}"

                "${PYTHON}" train.py \
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
                  --uncertainty_method "${UQ}" \
                  --gnn_backend "${GNN_BACKEND}" \
                  --ltau_bins "${LTAU_BINS}" \
                  --ltau_k "${LTAU_K}" \
                  --mc_samples "${MC_SAMPLES}" \
                  --dropout_rate "${DROPOUT_RATE}" \
                  --ensemble_size "${ENSEMBLE_SIZE}" \
                  --ensemble_seeds ${ENSEMBLE_SEEDS} \
                  --evidential_lambda "${EVIDENTIAL_LAMBDA}" \
                  --evidential_uncertainty "${EVIDENTIAL_UNCERTAINTY}" \
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
                  --head_seed "${HEAD_SEED}" \
                  2>&1 | tee -a "${LOG_FILE}"

                if [[ ! -f "${OUTPUT_DIR}/metrics.csv" ]]; then
                    echo "[FAIL] ${UQ_DIR} ${MOD_NAME} ${DATA_NAME}/${TARGET_COL}" >&2
                    failures=$((failures + 1))
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED" >> "${LOG_FILE}"
            done
        done
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done (failures: ${failures})"
exit $(( failures > 0 ))
