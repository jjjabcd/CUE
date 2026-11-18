"""CUE/CUE fusion training over the FP+GNN modality combination."""
import argparse

from cue.core.cue_runner import add_common_args, run_cue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p, default_modalities=["fp", "gnn"])
    return p.parse_args()


if __name__ == "__main__":
    run_cue(parse_args())
