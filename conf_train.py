"""
Train DownstreamMLP on precomputed Conformation (ConforFormer) embeddings.
Embeddings must be precomputed first (see scripts/run_prepare_feature.sh).
"""
import argparse

from cue.core.downstream_runner import add_common_args, run_downstream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_downstream(args, modality="conf")
