# Compound Scaling (복합 스케일링)

EfficientNet 논문(Tan & Le, 2019)에서 제안한 CNN 스케일링 방법입니다.

---

## 기존 단일 스케일링의 문제

기존에는 모델 성능을 높이기 위해 세 가지 차원 중 **하나**만 키웠습니다.

| 차원           | 방법                  | 한계                                           |
| -------------- | --------------------- | ---------------------------------------------- |
| **Width**      | 채널 수 증가          | 세밀한 특징 포착은 좋으나, 깊이 없이 한계 존재 |
| **Depth**      | 레이어 수 증가        | 깊어질수록 그래디언트 소실 문제                |
| **Resolution** | 입력 이미지 크기 증가 | 단독으로는 효율 낮음                           |

---

## Compound Scaling 핵심 아이디어

세 차원을 **균형 있게 동시에** 스케일링합니다.

$$
\text{depth}: d = \alpha^\phi \quad
\text{width}: w = \beta^\phi \quad
\text{resolution}: r = \gamma^\phi
$$

$$
\text{단, } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \geq 1,\ \beta \geq 1,\ \gamma \geq 1
$$

- $\phi$ (파이) : 사용 가능한 **컴퓨팅 자원**에 따라 조절하는 스케일링 계수
- $\alpha, \beta, \gamma$ : grid search로 찾은 최적 비율 (EfficientNet 기준: 1.2, 1.1, 1.15)
- FLOPs는 depth에 비례, width²·resolution²에 비례하므로 $\beta^2$, $\gamma^2$ 사용

---

## EfficientNet B0 ~ B7

| 모델 | $\phi$ | 입력 해상도 | 파라미터 수 |
| ---- | ------ | ----------- | ----------- |
| B0   | 0      | 224         | 5.3M        |
| B1   | 1      | 240         | 7.8M        |
| B4   | 4      | 380         | 19M         |
| B7   | 7      | 600         | 66M         |

> 프로젝트의 `efficiNetB4`는 $\phi=4$로 스케일업된 모델입니다.

---

## 핵심 장점

- 단일 차원 스케일링 대비 **같은 FLOPs에서 더 높은 정확도**
- 세 차원의 균형이 맞아 **과적합 없이 효율적** 스케일업 가능
- B0 하나만 설계하면 나머지는 $\phi$ 조절만으로 파생 가능
