"""
Train DownstreamMLP on precomputed GNN embeddings (MolE or MGSSL, selectable via
--gnn_backend). Embeddings must be precomputed first (see scripts/run_prepare_feature.sh
--modalities gnn gnn_mgssl).
"""
import argparse

from cue.core.downstream_runner import add_common_args, run_downstream

GNN_BACKEND_TO_MODALITY = {"mole": "gnn", "mgssl": "gnn_mgssl"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--gnn_backend", type=str, default="mgssl", choices=list(GNN_BACKEND_TO_MODALITY.keys()),
                    help="Which pretrained GNN embedding to train on (MolE or MGSSL).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_downstream(args, modality=GNN_BACKEND_TO_MODALITY[args.gnn_backend])
