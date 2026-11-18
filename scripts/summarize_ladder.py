"""Collect the modality-ladder results into one table.

Reads ``results/<UQ>/<DIS|INH>/<target>/v1/<combo>/metrics.csv`` (written by
scripts/run_modality_ladder.sh) and prints, per metric, a combo x UQ grid averaged over
the 8 CUE datasets, plus the per-dataset detail. Missing runs are reported, not hidden.

Usage:
    python scripts/summarize_ladder.py [--metric rmse] [--csv OUT.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

DATASETS = ("DIS", "INH")
TARGETS = ("Ssel", "PI", "WS2", "RS2")
METRICS = ("rmse", "r2", "pcc")
HIGHER_IS_BETTER = {"rmse": False, "r2": True, "pcc": True}

# Directory name -> label, in the order the ladder runs them.
UQ_METHODS = {"LTAU": "LTAU", "MC_Dropout": "MC Dropout",
              "Evidential": "Evidential", "Deep_Ensemble": "Deep Ensemble"}
COMBOS = ("FP+GNN", "FP+GNN+IMG", "FP+GNN+IMG+CONF", "FP+GNN+IMG+CONF+SMI")


def read_metrics(uq_dir: str, dataset: str, target: str, combo: str) -> dict:
    """Metric name -> value for the 'mlp' row, empty if the run is absent."""
    path = RESULTS / uq_dir / dataset / target / "v1" / combo / "metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    row = frame[frame["model_name"] == "mlp"]
    if row.empty:
        return {}
    return {m: float(row[m].values[0]) for m in METRICS if m in row.columns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default=None, choices=list(METRICS),
                        help="Show only this metric (default: all three).")
    parser.add_argument("--csv", default=None, help="Optional path to save the long table.")
    args = parser.parse_args()

    rows = []
    for uq_dir, uq_label in UQ_METHODS.items():
        for combo in COMBOS:
            for dataset in DATASETS:
                for target in TARGETS:
                    found = read_metrics(uq_dir, dataset, target, combo)
                    rows.append({
                        "UQ": uq_label, "Combo": combo, "Pair": f"{dataset}-{target}",
                        **{m: found.get(m, float("nan")) for m in METRICS},
                    })

    table = pd.DataFrame(rows)
    done = int(table["rmse"].notna().sum())
    print(f"완료: {done} / {len(table)} run\n")

    for metric in ([args.metric] if args.metric else METRICS):
        arrow = "↑" if HIGHER_IS_BETTER[metric] else "↓"
        grid = table.pivot_table(index="Combo", columns="UQ", values=metric, dropna=False)
        grid = grid.reindex(index=list(COMBOS), columns=list(UQ_METHODS.values()))
        print(f"=== {metric.upper()} {arrow} (8개 데이터셋 평균)")
        print(grid.round(4).to_string(), "\n")

    missing = table[table["rmse"].isna()]
    if not missing.empty:
        print("아직 없는 run:")
        for (uq, combo), grp in missing.groupby(["UQ", "Combo"]):
            print(f"  {uq:14} {combo:22} {len(grp)}개")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.round(6).to_csv(out, index=False)
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
