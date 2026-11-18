# 저장소 통합 기록 (2026-11-18)

`work/CUE/` 아래 네 저장소에 흩어져 있던 코드와 결과를 `Thesis/CUE` 한 곳으로 모았다.
이제 **CUE 관련 실험의 단일 출처는 이 디렉터리**다.

## 통합한 것

| 출처 | 가져온 것 | 비고 |
|---|---|---|
| `CUE/cue4modal` | `cue/` 패키지 135 py, 학습·사다리·융합 스크립트 | **기능적 상위집합** |
| `CUE/cue` | `notebook/` 47개, `scripts/` 26개, `train_{ltau,tau}_only.py` | 분석 주력 |
| `CUE/cue4unc` | UQ 실행 스크립트 10개, `time_predict.py` | 패키지는 중복 |
| `CUE/cue_concat` | `scripts/cue_concat_run_all.sh` | 패키지는 중복 |

### 패키지를 cue4modal 것으로 택한 근거

세 패키지를 모듈·함수 단위로 대조한 결과:

```
Thesis/CUE/cue   36 py   고유 모듈 0, 고유 def/class 0
CUE/cue/cue      40 py   고유 모듈 0, 고유 def/class 0
CUE/cue4unc/cue  38 py   고유 모듈 0
CUE/cue_concat   33 py   고유 모듈 0
CUE/cue4modal   135 py   ← 위 전부를 포함 + 95개 추가
```

공통 모듈 중 내용이 다른 5개도 전부 cue4modal 쪽이 확장판이었다
(`core/model.py` 371→583줄, `utils/FeatureExtraction.py` 106→373줄,
`core/trainer.py` 871→1226줄, `core/train_mlp.py` 207→241줄, `core/mlp.py` 51→66줄).
`cue` 쪽에만 있는 함수/클래스는 하나도 없었다.

cue4modal 에만 있던 것: `cue/mole`(MolE GNN), `cue/conformer`(3D), `cue/mgssl`,
`cue/core/evidential.py`.

### 결과적으로 이 패키지가 지원하는 것

```
모달리티  fp · img · gnn(MolE) · gnn_mgssl · conf · smi
UQ        ltau · mc_dropout · evidential · deep_ensemble
융합      softmax · uniform · random · sum · custom   (+ concat 은 model.py 분기)
```

## 디렉터리

```
Thesis/CUE/
  cue/                패키지 (135 py)
  notebook/           분석 노트북 47개 (tau, ltau_param, uncertainty, weight, emb, fusion, modal, benchmark, external)
  scripts/            실행 스크립트 55개 + 마이그레이션 스크립트
  results/            통합 결과 트리 — 규칙은 results/README.md
  scaffold_results/   scaffold split 24셀
  results_aux/        셀이 아닌 산출물 (analysis, plots, table, weight, davis_pred ...)
  data/               DIS/INH CSV
  thesis_eval/        TDC 평가
  *_train.py          모달리티별 학습 진입점 (gnn, conf, smi, FP_GNN, FP_IMG_GNN, IMG_GNN)
  train.py predict.py train_ltau_only.py train_tau_only.py time_predict.py
```

## 데이터 경로

스크립트 30여 개가 `./../chembl/data/selectivity_df/...` 를 쓴다. 데이터 원본은
`work/CUE/chembl` 에 그대로 두고, `work/Thesis/chembl -> ../CUE/chembl` 심볼릭 링크로
해결했다. **`CUE/chembl` 은 지우면 안 된다.**

마찬가지로 `CUE/dta_results`(4.2절 베이스라인)와 `CUE/qsar_results`(외부 검증)도
`Thesis/paper/notebook/{performance,external}.ipynb` 가 직접 읽으므로 유지해야 한다.

## 결과 트리 변경

버전 디렉터리(`v1`/`v2`/`v3`)를 없애고 모달리티 조합을 그 자리에 넣었다:

```
이전:  <ds>/<target>/v1/CUE                       (cue)
       <UQ>/<ds>/<target>/v1/CUE                  (cue4unc)
       <UQ>/<ds>/<target>/v1/<combo>              (cue4modal 사다리)
이후:  <UQ>/<ds>/<target>/<modality>/<variant>
```

928셀을 옮겼고 (`MIGRATION.tsv`), 옮긴 뒤 `Thesis/paper` 의 노트북 5개
(`performance`, `cue_variant_rank`, `ladder`, `modal`, `modal_table`)를 새 경로로 고쳐
재실행했다. **모든 수치가 이전과 동일**함을 확인했다.

## 알려진 이슈

- 2026-11-18 현재 호스트의 NVIDIA 드라이버가 불일치 상태다
  (커널 모듈 580.173.02 vs 라이브러리 580.178.04). 재부팅 전까지 GPU 를 쓸 수 없다.
  통합 작업 자체와는 무관하다.
- `environment.yml`(기존)과 `environment.cue4modal.yml`(cue4modal 것)이 둘 다 있다.
  통합 패키지는 후자의 의존성(torch_geometric, pytorch-lightning, DeBERTa, hydra,
  onnxruntime)을 필요로 한다.
