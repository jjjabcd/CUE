"""
Collect the Evidential Regression results into one table.

Reads ``results/Evidential/{DIS,INH}/{PI,Ssel,WS2,RS2}/v1/CUE/metrics.csv`` and
prints one row per dataset-target, alongside MC Dropout and Deep Ensemble where
those exist, so the three UQ methods can be compared at a glance.

Usage:
    python scripts/summarize_evidential.py [--csv OUT.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("DIS", "INH")
TARGETS = ("PI", "Ssel", "WS2", "RS2")
METRICS = ("rmse", "r2", "pcc")

# All three methods now live under the unified names
# (results/<UQ>/<DIS|INH>/<target>/v1/CUE/).
METHODS = {
    "Evidential": ("Evidential", "CUE"),
    "MC Dropout": ("MC_Dropout", "CUE"),
    "Deep Ensemble": ("Deep_Ensemble", "CUE"),
}


def read_metrics(method_dir: str, model_dir: str, dataset: str, target: str) -> dict:
    """
    Args:
        method_dir (str): Results subdirectory for the UQ method.
        model_dir (str): Innermost model directory name.
        dataset (str): Dataset name, 'DIS' or 'INH'.
        target (str): Target name, e.g. 'WS2'.

    Returns:
        dict: Metric name -> value for the 'mlp' row, empty if absent.
    """
    path = REPO_ROOT / "results" / method_dir / dataset / target / "v1" / model_dir / "metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    row = frame[frame["model_name"] == "mlp"]
    if row.empty:
        return {}
    return {m: float(row[m].values[0]) for m in METRICS if m in row.columns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Optional path to save the table.")
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        for target in TARGETS:
            entry = {"Dataset": dataset, "Target": target, "Pair": f"{dataset}-{target}"}
            for label, (method_dir, model_dir) in METHODS.items():
                found = read_metrics(method_dir, model_dir, dataset, target)
                for metric in METRICS:
                    entry[f"{label} {metric}"] = found.get(metric, float("nan"))
            rows.append(entry)

    table = pd.DataFrame(rows)
    missing = table["Evidential rmse"].isna().sum()

    print(table[["Pair"] + [f"Evidential {m}" for m in METRICS]].round(4).to_string(index=False))
    print(f"\nEvidential results found: {len(table) - missing}/{len(table)}")

    comparison = [c for c in table.columns if c.endswith(" rmse")]
    if len(comparison) > 1:
        print("\nRMSE by method:")
        print(table[["Pair"] + comparison].round(4).to_string(index=False))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.round(6).to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
