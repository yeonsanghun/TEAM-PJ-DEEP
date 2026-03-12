# convNext가 efficiNetB4보다 성능이 좋은 이유

> 기준: TensorBoard Loss/train · Loss/val 비교 (2026-03-10 실험)

---

## 수치 비교

| 모델 | Loss/train (Smoothed) | Loss/val (Smoothed) | 과적합 여부 |
|---|---|---|---|
| **convNext** | 0.0347 | 0.0292 | ✅ 없음 — train ≈ val |
| **resNet** | 0.013 | 0.0149 | ✅ 없음 — train ≈ val |
| **efficiNetB4** | 0.1224 | 0.0955 | ⚠️ 수렴 미완료 |

---

## 이유 1 — Fine-tuning 대상 레이어 수

| 모델 | Freeze | Unfreeze (학습 대상) |
|---|---|---|
| **convNext** | features[0~5] | **features[6] + features[7]** (2개 블록) |
| **efficiNetB4** | features[0~7] | **features[-1]** (1개 블록) |

convNext는 더 많은 레이어를 도메인에 맞게 재조정합니다.
efficiNetB4는 마지막 1개 블록만 학습하므로 전입신고서 도메인 적응이 제한적입니다.

---

## 이유 2 — 아키텍처 설계 철학의 차이

### convNext — "Transformer 개념을 CNN에 이식"

```
각 블록: Depthwise Conv(7×7) → LayerNorm → PointwiseConv → GELU
                   ↑                  ↑
           넓은 수용 범위         안정적 정규화
```

- 7×7 Depthwise Conv로 **더 넓은 영역을 한 번에 봄** → 양식 전체 레이아웃 파악에 유리
- LayerNorm이 배치 크기에 무관하게 안정적 → **소규모 데이터(649장)에서 강점**

### efficiNetB4 — "Compound Scaling으로 효율 최적화"

```
각 블록: MBConv (3×3 또는 5×5) → BatchNorm → Swish
                                      ↑
                              배치 통계에 의존
```

- BatchNorm은 배치 내 통계로 정규화 → **배치가 작을수록(4~8장) 통계 불안정**
- 649장 / batch_size=8 → 배치당 8장만으로 정규화 → 노이즈 증가

---

## 이유 3 — 사전학습 가중치의 전이 효과

$$\text{전이 효과} \propto \frac{\text{사전학습 도메인과의 유사성}}{\text{학습해야 할 파라미터 수}}$$

| | convNext-Small | EfficientNet-B4 |
|---|---|---|
| Unfreeze 파라미터 수 | 많음 (2블록 + head) | 적음 (1블록 + head) |
| 아키텍처 | 최신 (2022) | 구형 (2019) |
| ImageNet Top-1 | 83.1% | 83.0% |
| 소규모 fine-tuning | ✅ 강함 | ⚠️ BatchNorm 불리 |

---

## 이유 4 — 이번 실험에서의 직접 원인 요약

```
efficiNetB4 Loss/val이 높은 구체적 이유:

1. Optuna가 찾은 batch_size가 작았을 가능성
   → BN 통계 불안정 → 학습 노이즈 큼 (Loss/train 그래프의 진폭이 크게 나타남)

2. Fine-tuning 레이어 1개
   → 전입신고서 도메인 적응 범위가 좁음

3. EPOCHS=20으로 수렴 미완료
   → Loss/val 곡선이 epoch 20에서도 하강 중
```

---

## 결론

아키텍처 자체의 ImageNet 성능은 거의 동일(83%)하지만,
**소규모 데이터 + 문서 도메인 fine-tuning 환경**에서는
LayerNorm 기반의 convNext가 BatchNorm 기반 efficiNetB4보다 구조적으로 유리합니다.

| 핵심 차이 | convNext 유리한 이유 |
|---|---|
| 정규화 방식 | LayerNorm → 배치 크기 무관하게 안정적 |
| 수용 범위 | 7×7 커널 → 양식 레이아웃 전체 파악 |
| Fine-tuning 범위 | 2개 블록 unfreeze → 더 넓은 도메인 적응 |
| 아키텍처 세대 | 2022 설계 → Transformer 설계 원칙 반영 |
