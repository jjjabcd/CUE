#!/bin/bash
# Predict the DAVIS external set with every trained CUE uncertainty variant.
#
# The models were trained on the ChEMBL DIS folds; DAVIS is used only as an
# external test set, so nothing is retrained here. predict.py is called once per
# fold (--specific_fold) because the external-validation protocol in
# qsar_results/scripts/build_qsar_results.py computes metrics per fold and only
# then averages.
#
# Output: ./results/davis_pred/{MC_Dropout,Deep_Ensemble,Evidential}/<target>/fold<k>/pred.csv

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/cue/bin/python}"
DAVIS="../chembl/data/selectivity_df/davis_selectivity.csv"
DEVICE="cuda:0"
VERSION="v1"
FOLDS=(1 2 3 4 5)

METHODS=("MC_Dropout" "Deep_Ensemble" "Evidential")
DATA_NAME="DIS"
LEAF="CUE"

for NAME in "${METHODS[@]}"; do
    METHOD_DIR="${NAME}"

    for TARGET_COL in PI Ssel WS2 RS2; do
        TARGET="${TARGET_COL}"

        for FOLD in "${FOLDS[@]}"; do
            OUT_DIR="./results/davis_pred/${NAME}/${TARGET}/fold${FOLD}"
            mkdir -p "${OUT_DIR}"

            echo "[$(date '+%H:%M:%S')] ${NAME} ${TARGET} fold${FOLD}"
            "${PYTHON}" predict.py \
                --input_file "${DAVIS}" \
                --smiles_col SMILES \
                --base_dir "./results/${METHOD_DIR}" \
                --data_name "${DATA_NAME}" \
                --target_cols "${TARGET_COL}" \
                --version "${VERSION}" \
                --model_name "${LEAF}" \
                --device "${DEVICE}" \
                --batch_size 1024 \
                --specific_fold "fold${FOLD}" > /dev/null

            PRED_FILE="prediction_${DATA_NAME}_${TARGET_COL}_${VERSION}_${LEAF}.csv"
            if [[ ! -f "${PRED_FILE}" ]]; then
                echo "[ERROR] prediction file not produced: ${PRED_FILE}" >&2
                exit 1
            fi
            mv "${PRED_FILE}" "${OUT_DIR}/pred.csv"
        done
    done
done

echo "[$(date '+%H:%M:%S')] done"
