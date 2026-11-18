"""마이그레이션 2차: 1차에서 걸러지지 않은 나머지를 모두 Thesis/CUE 로 옮긴다.

1차(migrate_results_layout.py)는 DIS/INH x {PI,Ssel,WS2,RS2} 회귀 셀만 다뤘다. 남은 것들:

  1. 분류 타깃  Thesis/CUE/results/<ds>/S(100nM)_label|S(1uM)_label/v1/CUE
  2. MGSSL GNN  cue4modal/results/<ds>/<tg>/v2/GNN_MGSSL   (MolE 이전 백본)
  3. concat 전용 저장소  cue_concat/results/<ds>/<tg>/v1/FP+IMG
       -> 기존 FP+IMG/CONCAT 과 수치가 다른 별도 구현이라 CONCAT_BASELINE 으로 보존
  4. scaffold split  cue/scaffold_results, cue4unc/scaffold_results
       -> 분할 방식이 달라 results 와 섞지 않고 Thesis/CUE/scaffold_results 로 분리
  5. 셀이 아닌 산출물  cue/results/{analysis,davis_pred,plots,table,virtual_screening,weight},
       cue4unc/results/davis_pred  -> Thesis/CUE/results_aux/

이동만 한다. --apply 없이는 계획만 출력한다.
"""
from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple

WORK = "/home/rlawlsgurjh/work"
DST = os.path.join(WORK, "Thesis/CUE/results")
DST_SCAFFOLD = os.path.join(WORK, "Thesis/CUE/scaffold_results")
DST_AUX = os.path.join(WORK, "Thesis/CUE/results_aux")
SRC_CUE = os.path.join(WORK, "CUE/cue")
SRC_UNC = os.path.join(WORK, "CUE/cue4unc")
SRC_MODAL = os.path.join(WORK, "CUE/cue4modal")
SRC_CONCAT = os.path.join(WORK, "CUE/cue_concat")

DATASETS = ("DIS", "INH")
TARGETS = ("PI", "Ssel", "WS2", "RS2")
LABEL_TARGETS = ("S(100nM)_label", "S(1uM)_label")


def is_cell(p: str) -> bool:
    return (os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json"))
            and any(d.startswith("fold") for d in os.listdir(p)))


def plan() -> List[Tuple[str, str, str]]:
    """(설명, src, dst) 목록."""
    out: List[Tuple[str, str, str]] = []

    # 1. 분류 타깃 (1차에서 TARGETS 에 없어 남은 것)
    for ds in DATASETS:
        for tg in LABEL_TARGETS:
            for var in ("CUE", "FP", "IMG"):
                src = os.path.join(DST, ds, tg, "v1", var)
                if not is_cell(src):
                    continue
                mod, leaf = (var, "CUE") if var in ("FP", "IMG") else ("FP+IMG", "CUE")
                out.append(("분류 타깃", src, os.path.join(DST, "LTAU", ds, tg, mod, leaf)))

    # 2. MGSSL GNN (MolE 이전 백본)
    for ds in DATASETS:
        for tg in TARGETS:
            src = os.path.join(SRC_MODAL, "results", ds, tg, "v2", "GNN_MGSSL")
            if is_cell(src):
                out.append(("MGSSL GNN", src,
                            os.path.join(DST, "LTAU", ds, tg, "GNN_MGSSL", "CUE")))

    # 3. cue_concat 저장소 (별도 concat 구현)
    for ds in DATASETS:
        for tg in TARGETS:
            src = os.path.join(SRC_CONCAT, "results", ds, tg, "v1", "FP+IMG")
            if is_cell(src):
                out.append(("cue_concat", src,
                            os.path.join(DST, "LTAU", ds, tg, "FP+IMG", "CONCAT_BASELINE")))

    # 4. scaffold split
    for ds in DATASETS:
        for tg in TARGETS:
            src = os.path.join(SRC_CUE, "scaffold_results", ds, tg, "v1", "CUE")
            if is_cell(src):
                out.append(("scaffold/LTAU", src,
                            os.path.join(DST_SCAFFOLD, "LTAU", ds, tg, "FP+IMG", "CUE")))
    for uq in ("MC_Dropout", "Evidential", "Deep_Ensemble"):
        for ds in DATASETS:
            for tg in TARGETS:
                src = os.path.join(SRC_UNC, "scaffold_results", uq, ds, tg, "v1", "CUE")
                if is_cell(src):
                    out.append((f"scaffold/{uq}", src,
                                os.path.join(DST_SCAFFOLD, uq, ds, tg, "FP+IMG", "CUE")))

    # 5. 셀이 아닌 산출물
    for name in ("analysis", "davis_pred", "plots", "table", "virtual_screening", "weight"):
        src = os.path.join(SRC_CUE, "results", name)
        if os.path.isdir(src):
            out.append(("aux/cue", src, os.path.join(DST_AUX, "cue", name)))
    src = os.path.join(SRC_UNC, "results", "davis_pred")
    if os.path.isdir(src):
        out.append(("aux/cue4unc", src, os.path.join(DST_AUX, "cue4unc", "davis_pred")))
    src = os.path.join(SRC_CUE, "scaffold_results", "plots")
    if os.path.isdir(src):
        out.append(("aux/scaffold", src, os.path.join(DST_AUX, "cue", "scaffold_plots")))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    moves = plan()
    by_kind: dict = {}
    for kind, s, d in moves:
        by_kind.setdefault(kind.split("/")[0], []).append((s, d))
    for kind, mv in by_kind.items():
        print(f"[{kind}] {len(mv)}")

    dsts = [d for _, _, d in moves]
    dupes = {d for d in dsts if dsts.count(d) > 1}
    exists = [d for d in dsts if os.path.exists(d)]
    if dupes or exists:
        print("\n[ERROR] 목적지 충돌:")
        for d in list(dupes)[:5] + exists[:5]:
            print("  ", d)
        raise SystemExit(1)

    print(f"\n총 {len(moves)}건")
    for kind, s, d in moves[:12]:
        print(f"  [{kind}] {os.path.relpath(s, WORK)}\n      -> {os.path.relpath(d, WORK)}")

    if not args.apply:
        print("\n(계획만. 실제 이동은 --apply)")
        return
    for kind, s, d in moves:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.move(s, d)
    print(f"완료: {len(moves)}건 이동")


if __name__ == "__main__":
    main()
