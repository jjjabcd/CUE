# LTAU Uncertainty = 0 현상 원인 분석

## 증상

`LTAU_BINS`가 작은 그리드 포인트(예: `LTAU_BINS=5`)에서 `ltau/{fp,img}/{train,val,test}_uncertainty.csv`의
`uncertainty` 컬럼이 정확히 `0`으로 찍히는 샘플이 다수 발견됨. 반면 `LTAU_BINS=5000`처럼 bin 수가 많은 소스
런(`CUE` 폴더)에서는 0이 전혀 관측되지 않음.

예시 (`results/DIS/WS2/v1/LTAU_BIN5_K3/fold1/ltau/fp/train_uncertainty.csv`):
train 399개 중 56개가 `uncertainty == 0`.

## 계산 방식

LTAU uncertainty(entropy 모드)는 `cue/ltau/descriptor.py`에서 다음 순서로 계산된다.

1. 각 화합물의 학습 epoch별 error(`error_trajectory`)를 모아, 전체 데이터셋 기준
   `[0, max_error * 1.2]` 구간을 `num_bins`(=`LTAU_BINS`)개 동일 폭 구간으로 나눈다
   (`make_shared_bin_edges`).
2. 화합물별 error trajectory를 이 bin들에 대해 히스토그램화해 확률분포(PDF)로 만든다
   (`trajectories_to_pdfs`).
3. 이 분포의 **Shannon entropy**를 `log(num_bins)`로 정규화해 `uncertainty ∈ [0, 1]`로 반환한다
   (`pdfs_to_uncertainty_entropy`).

```python
ent = -(pmass * np.log(np.maximum(pmass, eps))).sum(axis=1)
ent_max = np.log(pmass.shape[1] + eps)
u = ent / np.maximum(ent_max, eps)
```

분포가 한 bin에 확률질량 100%가 몰린(one-hot) 상태면 `entropy = -1*log(1) = 0` 이므로
`uncertainty = 0`이 정확히 나온다. 이는 버그가 아니라 entropy 정의상 당연한 결과다.

## 원인: bin 폭 대비 error 변동폭이 작음

`LTAU_BINS`가 작을수록 bin 폭이 커진다. 예를 들어 `num_bins=5`, 데이터셋 전체 최대 error가
0.845 근처라면:

```
bin edges = [0, 0.2029, 0.4057, 0.6086, 0.8115, 1.0144]   (폭 ≈ 0.203)
```

한 화합물의 학습 전 구간 error가 예를 들어 `min=6.5e-6, max=0.169`처럼 이 bin 폭(0.203)보다
작은 범위 안에서만 움직이면, 그 화합물의 모든 epoch error가 **bin 0 하나에만** 들어간다.

```
histogram = [1.0, 0, 0, 0, 0]   # 확률질량이 한 bin에 100% 집중
entropy   = 0
```

실제 검증 (`LTAU_BIN5_K3/fold1`, fp):

```
zero-uncertainty 화합물 trajectory: min=6.55e-06, max=0.1685
bin edges (num_bins=5): [0.0, 0.2029, 0.4057, 0.6086, 0.8115, 1.0144]
→ 모든 epoch error가 bin 0 안에 위치 → entropy = 0
```

반대로 `LTAU_BINS=5000`처럼 bin 수가 많으면 폭이 매우 좁아져(≈0.0002), 같은 화합물이라도
epoch마다 error가 조금만 변해도 여러 bin에 걸치게 되어 entropy가 0이 되는 경우가 사실상 없다.

## 결론

- **버그 아님.** entropy 기반 uncertainty가 "분포가 한 bin에 완전히 몰리면 0"이라는 정의를
  그대로 따른 결과.
- **원인은 해상도(bin 개수) 부족.** `LTAU_BINS`가 작을수록(특히 `5`) 개별 화합물의 실제 error
  변동이 있어도 같은 bin에 뭉개져 "확실하다(uncertainty=0)"로 잘못 뭉뚱그려지는 화합물이
  많아진다.
- LTAU 그리드에 `5`(너무 coarse)와 `50000`(너무 fine, 반대로 bin이 대부분 비어 entropy가
  과도하게 커질 수 있음)처럼 극단값을 포함시킨 것은 이런 해상도 민감도를 확인하려는 의도로
  보이며, `LTAU_BINS=5` 결과를 분석할 때는 uncertainty 해상도 부족 아티팩트를 감안해서
  해석해야 한다.
