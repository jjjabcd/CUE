#!/bin/bash
# Fusion-policy ablation on the modality ladder.
#
# run_modality_ladder.sh 와 모든 설정이 같고 --fusion_policy 만 바꾼다. 4.3절의 진단에서
# Deep Ensemble / MC Dropout 의 softmax 가중치가 사실상 균등(모달리티 간 평균 가중치 차이
# 0.004)으로 나왔으므로, "불확실성 가중 융합이 균등 융합과 구별되는가"를 직접 검증한다.
#
#   uniform : 가중치를 1/M 로 고정 (불확실성을 전혀 쓰지 않는 기준선)
#   concat  : 가중치 없이 임베딩을 이어 붙임 (헤드 입력 차원이 M*D 로 커진다)
#
# 출력: ./results/<UQ>/<DIS|INH>/<target>/<uniform|concat>/<FP+GNN|...>
#   기존 사다리는 .../v1/... 이므로 서로 덮어쓰지 않는다.
#
# Usage:
#   FUSION_POLICY=uniform bash scripts/run_fusion_ablation.sh
#   FUSION_POLICY=concat  bash scripts/run_fusion_ablation.sh
#   FUSION_POLICY=uniform UQ_METHODS="ltau mc_dropout evidential deep_ensemble" \
#       bash scripts/run_fusion_ablation.sh          # 네 방법 전부
#   DRY_RUN=1 FUSION_POLICY=uniform bash scripts/run_fusion_ablation.sh
#
# 기본 UQ_METHODS 가 ltau 하나인 이유: uniform/concat 에서는 불확실성이 융합에 쓰이지 않으므로
# LTAU(표준 MSE 학습, dropout 0) 가 곧 "UQ 없는" 순수 기준선이다. 나머지 세 방법은 인코더
# 학습 자체가 달라서(evidential loss / dropout 0.2 / 10-member ensemble) 별도 셀이 되지만,
# 먼저 LTAU 32셀로 결론이 나는지 보는 편이 싸다.

set -uo pipefail

cd "$(dirname "$0")/.."

# 출력 버전 = 융합 정책 이름. 기존 사다리(v1, softmax)를 건드리지 않고 옆에 쌓인다.
FUSION_POLICY="${FUSION_POLICY:-uniform}"
VERSION="${VERSION:-${FUSION_POLICY}}"
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
UQ_METHODS="${UQ_METHODS:-ltau}"

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
