#!/bin/bash

# =============================
# Settings
# =============================
DEVICE="cuda:0"
DATA_DIR="../chembl/data/selectivity_df"

# Optional filters
DATASET=""            # on-disk assay folder: "Kd" (=DIS) or "Ki" (=INH); empty = both
MODALITIES="fp img smi gnn gnn_mgssl conf"
OVERWRITE="0"          # set to "1" to re-extract existing .npy files
SKIP_KEYS="1"          # set to "1" to only compute missing feature keys

# =============================
# Run
# =============================
python -m cue.utils.prepare_features \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --modalities ${MODALITIES} \
    ${DATASET:+--dataset "${DATASET}"} \
    $( [ "${OVERWRITE}" = "1" ] && echo "--overwrite" ) \
    $( [ "${SKIP_KEYS}" = "1" ] && echo "--skip_existing_keys" )
