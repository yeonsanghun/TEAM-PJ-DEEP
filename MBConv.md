# MBConv (Mobile Inverted Bottleneck Convolution)

MobileNetV2에서 처음 제안된 블록으로, EfficientNet 등 경량 CNN 아키텍처의 핵심 구성 요소입니다.

---

## 구조 (3단계)

```
입력
 ↓
[1] Pointwise Conv (1×1) — 채널 확장 (Expansion)
 ↓
[2] Depthwise Conv (3×3 또는 5×5) — 공간 특징 추출
 ↓
[3] Pointwise Conv (1×1) — 채널 압축 (Projection)
 ↓
출력 (+ Skip Connection)
```

---

## 핵심 개념

| 개념                         | 설명                                                                       |
| ---------------------------- | -------------------------------------------------------------------------- |
| **Inverted Bottleneck**      | 일반 병목(bottleneck)과 반대로, 중간 레이어의 채널을 *확장*한 뒤 다시 줄임 |
| **Depthwise Separable Conv** | 채널별로 따로 conv → 파라미터 수 대폭 감소                                 |
| **Skip Connection**          | 입출력 채널이 같을 때 잔차 연결 적용 (ResNet 스타일)                       |
| **Expansion Factor**         | 채널을 몇 배로 확장할지 결정 (보통 6배)                                    |

---

## 일반 병목 vs MBConv 비교

```
일반 Bottleneck:  넓음 → 좁음 → 넓음  (압축 후 처리)
MBConv:           좁음 → 넓음 → 좁음  (확장 후 처리)
```

---

## 장점

- 파라미터 수와 연산량(FLOPs)이 적어 **모바일/경량 환경**에 적합
- Depthwise Conv로 공간 정보를 효율적으로 처리
- Skip Connection으로 **그래디언트 소실 방지**

---

## 사용 아키텍처

- **MobileNetV2/V3**
- **EfficientNet (B0~B7)** — MBConv + SE(Squeeze-and-Excitation) 블록 조합
- **EfficientNetV2** — MBConv + Fused-MBConv 혼용

> 현재 프로젝트의 `efficiNetB4`도 EfficientNet-B4 기반이므로, 내부적으로 MBConv 블록이 반복적으로 쌓인 구조입니다.
