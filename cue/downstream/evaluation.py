import numpy as np
import pandas as pd
from cue.utils.metric import get_metrics


def aggregate_fold_metrics(df):
    """
    Aggregate metrics across folds.
    """
    results = []
    model_names = [c for c in df.columns if c not in ["GroundTruth", "Fold"]]

    for model in model_names:
        rmses, r2s, pccs = [], [], []
        for fold in df["Fold"].unique():
            sub = df[df["Fold"] == fold]
            met = get_metrics(sub["GroundTruth"], sub[model])
            rmses.append(met["rmse"])
            r2s.append(met["r2"])
            pccs.append(met["pcc"])

        results.append({
            "model_name": model,
            "rmse": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "r2": np.mean(r2s),
            "r2_std": np.std(r2s),
            "pcc": np.mean(pccs),
            "pcc_std": np.std(pccs),
        })

    return pd.DataFrame(results)
