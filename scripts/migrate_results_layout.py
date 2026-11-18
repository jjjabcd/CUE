"""Thesis/CUE/results 를 통합 레이아웃으로 재배치한다.

목표 레이아웃
-------------
    Thesis/CUE/results/<UQ>/<DIS|INH>/<target>/<MODALITY>/<VARIANT>

  UQ        LTAU | MC_Dropout | Evidential | Deep_Ensemble
  MODALITY  FP | IMG | GNN | CONF | SMI | FP+IMG | FP+GNN | FP+GNN+IMG | ...
  VARIANT   CUE(기본=softmax 융합) | UNIFORM | CONCAT | SUM | TAU* | LTAU_BIN* | R1..R30 | CUSTOM_*

버전 디렉터리(v1/v2/v3)는 사라지고, 그 자리에 모달리티 조합이 들어간다. 버전이 실제로
구분하던 것(드롭아웃 스윕, 융합 정책)은 VARIANT 이름으로 옮긴다.

이동만 한다(같은 파일시스템이면 즉시). --apply 없이는 계획만 출력한다.

사용법
    python3 migrate_results_layout.py              # 계획만 출력
    python3 migrate_results_layout.py --apply      # 실제 이동
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from typing import List, Tuple

WORK = "/home/rlawlsgurjh/work"
DST_ROOT = os.path.join(WORK, "Thesis/CUE/results")
SRC_UNC = os.path.join(WORK, "CUE/cue4unc/results")
SRC_MODAL = os.path.join(WORK, "CUE/cue4modal/results")

DATASETS = ("DIS", "INH")
TARGETS = ("PI", "Ssel", "WS2", "RS2")
UQ_DIRS = ("LTAU", "MC_Dropout", "Evidential", "Deep_Ensemble")

# cue4modal 의 버전 디렉터리 -> VARIANT 이름
MODAL_VERSION_TO_VARIANT = {
    "v1": "CUE",                    # 기본: 불확실성 가중(softmax) 융합
    "uniform": "UNIFORM",
    "concat": "CONCAT",
    "uniform_refit": "UNIFORM_REFIT",  # refit 경로 검증용(전체 재학습본과 대조)
}


def is_cell(path: str) -> bool:
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, "config.json"))
            and any(d.startswith("fold") for d in os.listdir(path)))


def plan_thesis_cue() -> List[Tuple[str, str]]:
    """Thesis/CUE/results/<ds>/<tg>/v1/<VAR>  ->  LTAU/<ds>/<tg>/<MOD>/<VAR>

    이 트리는 전부 LTAU + FP/IMG 계열이다. 단일 모달(FP, IMG)만 모달리티 이름이
    곧 변종 이름이므로 leaf 를 CUE 로 통일해 깊이를 맞춘다."""
    moves = []
    for ds in DATASETS:
        for tg in TARGETS:
            v1 = os.path.join(DST_ROOT, ds, tg, "v1")
            if not os.path.isdir(v1):
                continue
            for var in sorted(os.listdir(v1)):
                src = os.path.join(v1, var)
                if not is_cell(src):
                    continue
                if var in ("FP", "IMG"):
                    mod, leaf = var, "CUE"
                else:
                    mod, leaf = "FP+IMG", var
                moves.append((src, os.path.join(DST_ROOT, "LTAU", ds, tg, mod, leaf)))
    return moves


def plan_cue4unc() -> List[Tuple[str, str]]:
    """cue4unc/results/<UQ>/<ds>/<tg>/<ver>/CUE -> <UQ>/<ds>/<tg>/FP+IMG/<VARIANT>

    MC_Dropout 의 v2/v3 는 드롭아웃 스윕이므로 config 에서 값을 읽어 이름을 만든다."""
    moves = []
    for uq in ("MC_Dropout", "Evidential", "Deep_Ensemble"):
        base = os.path.join(SRC_UNC, uq)
        if not os.path.isdir(base):
            continue
        for ds in DATASETS:
            for tg in TARGETS:
                tgt_dir = os.path.join(base, ds, tg)
                if not os.path.isdir(tgt_dir):
                    continue
                for ver in sorted(os.listdir(tgt_dir)):
                    src = os.path.join(tgt_dir, ver, "CUE")
                    if not is_cell(src):
                        continue
                    if ver == "v1":
                        leaf = "CUE"
                    else:
                        cfg = json.load(open(os.path.join(src, "config.json")))
                        dr = cfg.get("dropout_rate")
                        leaf = f"DROPOUT{dr}" if dr is not None else ver.upper()
                    moves.append((src, os.path.join(DST_ROOT, uq, ds, tg, "FP+IMG", leaf)))
    return moves


def plan_cue4modal_ladder() -> List[Tuple[str, str]]:
    """cue4modal/results/<UQ>/<ds>/<tg>/<ver>/<COMBO> -> <UQ>/<ds>/<tg>/<COMBO>/<VARIANT>"""
    moves = []
    for uq in UQ_DIRS:
        base = os.path.join(SRC_MODAL, uq)
        if not os.path.isdir(base):
            continue
        for ds in DATASETS:
            for tg in TARGETS:
                tgt_dir = os.path.join(base, ds, tg)
                if not os.path.isdir(tgt_dir):
                    continue
                for ver in sorted(os.listdir(tgt_dir)):
                    leaf = MODAL_VERSION_TO_VARIANT.get(ver)
                    if leaf is None:
                        print(f"  [WARN] 알 수 없는 버전 디렉터리, 건너뜀: "
                              f"{os.path.join(tgt_dir, ver)}", file=sys.stderr)
                        continue
                    for combo in sorted(os.listdir(os.path.join(tgt_dir, ver))):
                        src = os.path.join(tgt_dir, ver, combo)
                        if not is_cell(src):
                            continue
                        moves.append((src, os.path.join(DST_ROOT, uq, ds, tg, combo, leaf)))
    return moves


def plan_cue4modal_unimodal() -> List[Tuple[str, str]]:
    """cue4modal/results/<ds>/<tg>/v1/<MOD> -> LTAU/<ds>/<tg>/<MOD>/<VARIANT>

    이 트리는 단일 모달(GNN/CONF/SMI) 기준선과, 사다리 이전에 돌린 몇몇 조합이다.
    FP+GNN 은 사다리 쪽(head_seed=1 로 다시 맞춘 것)과 경로가 겹치면서 수치가 다르므로
    (예: DIS-PI 0.2484 vs 0.2427) 덮어쓰지 않고 CUE_PRE_HEADSEED 로 따로 보존한다."""
    moves = []
    for ds in DATASETS:
        for tg in TARGETS:
            v1 = os.path.join(SRC_MODAL, ds, tg, "v1")
            if not os.path.isdir(v1):
                continue
            for mod in sorted(os.listdir(v1)):
                src = os.path.join(v1, mod)
                if not is_cell(src):
                    continue
                leaf = "CUE_PRE_HEADSEED" if mod == "FP+GNN" else "CUE"
                moves.append((src, os.path.join(DST_ROOT, "LTAU", ds, tg, mod, leaf)))
    return moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 이동한다")
    ap.add_argument("--manifest", default=os.path.join(DST_ROOT, "..", "MIGRATION.tsv"))
    args = ap.parse_args()

    groups = [
        ("Thesis/CUE (FP+IMG 계열, LTAU)", plan_thesis_cue()),
        ("cue4unc (FP+IMG, 다른 UQ)", plan_cue4unc()),
        ("cue4modal 사다리", plan_cue4modal_ladder()),
        ("cue4modal 단일모달/구버전", plan_cue4modal_unimodal()),
    ]

    all_moves: List[Tuple[str, str]] = []
    for name, mv in groups:
        print(f"[{name}] {len(mv)} 셀")
        all_moves.extend(mv)

    dst_count = Counter(d for _, d in all_moves)
    dupes = {d: n for d, n in dst_count.items() if n > 1}
    if dupes:
        print("\n[ERROR] 목적지가 겹칩니다. 중단합니다:")
        for d, n in list(dupes.items())[:10]:
            print(f"  {n}x  {d}")
        raise SystemExit(1)

    existing = [d for _, d in all_moves if os.path.exists(d)]
    if existing:
        print(f"\n[ERROR] 이미 존재하는 목적지 {len(existing)}개. 중단합니다:")
        for d in existing[:10]:
            print(f"  {d}")
        raise SystemExit(1)

    print(f"\n총 {len(all_moves)} 셀 이동 예정")
    print("\n예시 20개:")
    for s, d in all_moves[:20]:
        print(f"  {os.path.relpath(s, WORK)}\n    -> {os.path.relpath(d, WORK)}")

    with open(args.manifest, "w", encoding="utf-8") as f:
        f.write("src\tdst\n")
        for s, d in all_moves:
            f.write(f"{s}\t{d}\n")
    print(f"\n매니페스트 -> {os.path.normpath(args.manifest)}")

    if not args.apply:
        print("\n(계획만 출력했습니다. 실제 이동은 --apply)")
        return

    for i, (s, d) in enumerate(all_moves, 1):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.move(s, d)
        if i % 100 == 0:
            print(f"  {i}/{len(all_moves)}", flush=True)
    print(f"완료: {len(all_moves)} 셀 이동")


if __name__ == "__main__":
    main()
