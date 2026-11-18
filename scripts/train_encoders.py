"""Stage 1 — 모달리티 MLP 10개(seed 0..9)를 데이터셋 x 타깃 x fold x 모달리티 마다 학습한다.

왜 이 스크립트가 필요한가
-------------------------
기존 train.py 는 (UQ 방법 x 모달리티 조합) 셀마다 인코더를 처음부터 다시 학습했다. 그래서
LTAU / MC Dropout / Deep Ensemble 이 서로 다른 인코더를 쓰고 있었고(체크포인트 가중치 전부
불일치), "융합 정책만 다른 비교"가 성립하지 않았다.

이 스크립트는 인코더를 **한 번만** 학습해 공유한다:

    m0            -> LTAU(학습 궤적으로 추정) / MC Dropout(dropout 켠 forward) / DE 멤버 0
    m1..m9        -> Deep Ensemble 멤버 1..9

세 방법은 인코더가 같고 불확실성 u 를 내는 방식만 다르다. 융합에 쓰는 임베딩도 항상 멤버 0
에서 나오므로(cue/core/model.py predict) LTAU/MC Dropout/DE 의 임베딩이 완전히 같아진다.

dropout 은 10개 전부 0.2 로 통일한다 (MC Dropout 이 같은 MLP 를 써야 하므로).

규모: 8 dataset x 5 fold x 5 modality x 10 member = 2000 MLP.
모달리티 입력은 fold 의 .npy 캐시에서 읽으므로 인코더 추론 재계산은 없다.

출력 (MMGW/mmgw/results 레이아웃)
---------------------------------
    results_shared/<DATASET>/<TARGET>/modality_mlps/fold<k>/
        checkpoints/mlp_<mod>_m<i>.pth · checkpoints/scaler_<mod>.pkl
        history_<mod>_m<i>.csv
        trajectories/<mod>_m0/{train,val}_trajectories.csv      (m0 만 — LTAU 용)
        meta.json

Run
---
    ~/miniconda3/envs/cue/bin/python scripts/train_encoders.py --datasets DIS --targets PI
    ~/miniconda3/envs/cue/bin/python scripts/train_encoders.py            # 전체 2000개
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cue.core.mlp import MLP  # noqa: E402
from cue.core.model import (  # noqa: E402
    GNN_BACKEND_CACHE_KEY,
    MODALITY_CACHE_KEY,
)
from cue.core.downstream_runner import load_embedding_lookup  # noqa: E402
from cue.core.train_mlp import MLPTrainConfigs, train_mlp  # noqa: E402
from cue.core.trainer import Trainer  # noqa: E402
from cue.data.normalization import LabelNormalizer  # noqa: E402
from cue.utils.seed import seed  # noqa: E402
from cue.utils.utils import load_fold  # noqa: E402

# 기존 run_modality_ladder.sh / run_fusion_ablation.sh 와 같은 값.
DATA_SRC = {"DIS": "Kd", "INH": "Ki"}
TARGETS = ["PI", "Ssel", "WS2", "RS2"]
MODALITIES = ["fp", "gnn", "img", "conf", "smi"]
GNN_BACKEND = "mole"

HIDDEN_DIM = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
SAVE_EVERY = 1
DROPOUT_RATE = 0.2          # 10 멤버 전부 동일 (MC Dropout 이 m0 를 그대로 쓴다)
N_MEMBERS = 10
LABEL_MODE = "none"


def _cache_key(mod: str) -> str:
    return GNN_BACKEND_CACHE_KEY[GNN_BACKEND] if mod == "gnn" else MODALITY_CACHE_KEY[mod]


def fold_dir(out_root: str, ds: str, target: str, fold: str) -> str:
    return os.path.join(out_root, ds, target, "modality_mlps", fold)


def ckpt_path(out_root: str, ds: str, target: str, fold: str, mod: str, i: int) -> str:
    return os.path.join(fold_dir(out_root, ds, target, fold), "checkpoints",
                        f"mlp_{mod}_m{i}.pth")


def train_cell(ds: str, target: str, fold: str, mod: str, out_root: str, data_root: str,
               device: str, force: bool) -> Dict[str, float]:
    """한 (dataset, target, fold, modality) 의 멤버 m0..m9 를 학습한다."""
    fdir = fold_dir(out_root, ds, target, fold)
    os.makedirs(os.path.join(fdir, "checkpoints"), exist_ok=True)

    todo = [i for i in range(N_MEMBERS)
            if force or not os.path.exists(ckpt_path(out_root, ds, target, fold, mod, i))]
    if not todo:
        return {}

    fold_path = os.path.join(data_root, DATA_SRC[ds], fold)
    tr_id, va_id, _, X_tr, y_tr, X_va, y_va, _, _ = load_fold(fold_path, target)

    # 라벨 정규화는 셀 전체에서 하나 (train 으로만 fit) — cue_runner 와 같은 순서.
    lab = LabelNormalizer(LABEL_MODE)
    lab.fit(y_tr)
    y_tr_n = lab.transform(y_tr).reshape(-1, 1)
    y_va_n = lab.transform(y_va).reshape(-1, 1)

    lookup = load_embedding_lookup(fold_path, _cache_key(mod))
    get_features = lambda s: lookup[s]  # noqa: E731
    input_dim = int(np.asarray(next(iter(lookup.values()))).shape[-1])

    losses: Dict[str, float] = {}
    for i in todo:
        # 멤버마다 재시딩 후 모델을 새로 만든다: 가중치 초기화와 DataLoader 셔플 순서가 모두 달라진다.
        seed(i)
        trainer = Trainer(MLP(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE), device)

        cfg = MLPTrainConfigs(
            feature_name=mod,
            trainer=trainer,
            scaler=MinMaxScaler(),
            get_features=get_features,
            fold_dir=fdir,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            save_every=SAVE_EVERY,
            task_type="regression",
            # LTAU 는 m0 의 궤적만 쓴다. 나머지 9개는 저장하지 않아 디스크를 아낀다.
            save_trajectory=(i == 0),
            member_idx=i,
            num_members=N_MEMBERS,
        )
        t0 = time.time()
        losses[f"m{i}"] = train_mlp(tr_id, va_id, X_tr, X_va, y_tr_n, y_va_n, cfg)
        print(f"    {mod}_m{i}: val {losses[f'm{i}']:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    meta_path = os.path.join(fdir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"dataset": ds, "target": target, "fold": fold,
                 "gnn_backend": GNN_BACKEND, "hidden_dim": HIDDEN_DIM,
                 "dropout_rate": DROPOUT_RATE, "num_epochs": NUM_EPOCHS,
                 "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                 "label_mode": LABEL_MODE, "n_members": N_MEMBERS})
    meta.setdefault("modalities", {})[mod] = {"input_dim": input_dim, "val_loss": losses}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return losses


def main() -> None:
    global NUM_EPOCHS, N_MEMBERS
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=os.path.join(REPO, "results_shared"))
    p.add_argument("--data_root", default=os.path.join(
        os.path.dirname(REPO), "chembl", "data", "selectivity_df"))
    p.add_argument("--datasets", nargs="+", default=list(DATA_SRC))
    p.add_argument("--targets", nargs="+", default=TARGETS)
    p.add_argument("--modalities", nargs="+", default=MODALITIES)
    p.add_argument("--folds", nargs="+", default=[f"fold{k}" for k in range(1, 6)])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    # 스모크 테스트용. 기본값은 위 상수 그대로이므로 본 실행에서는 건드리지 않는다.
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--n_members", type=int, default=N_MEMBERS)
    a = p.parse_args()
    NUM_EPOCHS, N_MEMBERS = a.num_epochs, a.n_members

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — refusing to train 2000 MLPs on CPU.")

    cells = [(ds, t, f, m) for ds in a.datasets for t in a.targets
             for f in a.folds for m in a.modalities]
    print(f"[plan] {len(cells)} cells x {N_MEMBERS} members = {len(cells) * N_MEMBERS} MLPs",
          flush=True)

    t_start = time.time()
    for i, (ds, t, f, m) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {ds}/{t}/{f}/{m}  (+{time.time() - t_start:.0f}s)", flush=True)
        train_cell(ds, t, f, m, a.out_root, a.data_root, a.device, a.force)
    print(f"[done] {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
