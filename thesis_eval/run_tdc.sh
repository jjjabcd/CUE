#!/usr/bin/env bash
# CUE on all 7 TDC regression tasks, 5 folds each, one fold at a time. Finished folds are skipped.
cd "$(dirname "$0")/.."
PY=/home/rlawlsgurjh/miniconda3/envs/cue/bin/python
for d in Caco2_Wang Lipophilicity_AstraZeneca VDss_Lombardo Half_Life_Obach Clearance_Hepatocyte_AZ Clearance_Microsome_AZ LD50_Zhu; do
  $PY thesis_eval/run_cue_tdc.py --dataset $d --folds 1 2 3 4 5 || echo "FAILED tdc $d"
done
echo TDC_DONE
