# ResNet-50 Multi-Label Classification — Workflow

## 개요

`data2/` 디렉토리의 전입신고서 이미지에 대해 14개 필드 기입 여부를 판별하는 **Multi-Label Classification** 파이프라인.
ResNet-50 (pretrained `IMAGENET1K_V1`) 기반으로 Optuna 하이퍼파라미터 최적화와 GradCAM 시각화를 포함한다.

---

## 데이터 구조

| Split | 이미지 경로    | 레이블 파일              | 샘플 수 |
| ----- | -------------- | ------------------------ | ------- |
| Train | `data2/train/` | `data2/train_labels.csv` | 2,799   |
| Val   | `data2/val/`   | `data2/val_labels.csv`   | 599     |
| Test  | `data2/test/`  | `data2/test_labels.csv`  | 601     |

- **이미지 형식**: JPG, 플랫 디렉토리 구조 (클래스별 서브폴더 없음)
- **CSV 구조**: `filename` 컬럼 + 14개 이진(0/1) 레이블 컬럼
- **레이블 대상 (14개)**: 전입자*성명, 전입자*주민등록번호, 전입자*연락처, 전입자*서명도장, 전*시도, 전*시군구, 현*세대주성명, 현*연락처, 현*서명도장, 현*주소, 전입사유*체크, 우편물서비스*동의체크, 신청인*성명, 신청인*서명도장

---

## ResNet-50 아키텍처

```
conv1        →  7×7 Conv, stride 2          224×224 → 112×112
bn1 / relu   →  BatchNorm + ReLU
maxpool      →  3×3 MaxPool, stride 2       112×112 → 56×56
layer1       →  Bottleneck × 3              56×56
layer2       →  Bottleneck × 4              28×28
layer3       →  Bottleneck × 6              14×14
layer4       →  Bottleneck × 3              7×7        ← Fine-Tuning 대상
avgpool      →  Global Average Pooling      7×7 → 1×1
fc           →  Linear(2048, 14)            ← 교체 대상
```

- 파라미터 수: 약 **25M** (ImageNet Top-1 Acc. 76.1%)
- Bottleneck 블록 구조: `1×1 Conv → 3×3 Conv → 1×1 Conv` (채널 축소 후 확장)

---

## EfficientNetB4 vs ResNet-50 비교

| 항목           | EfficientNetB4                                             | ResNet-50                                    |
| -------------- | ---------------------------------------------------------- | -------------------------------------------- |
| 모델 로드      | `efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)` | `resnet50(weights=ResNet50_Weights.DEFAULT)` |
| Fine-tune 대상 | `features[-1]` (마지막 MBConv 블록)                        | `layer4` (마지막 Bottleneck 그룹)            |
| 헤드 교체 위치 | `classifier[1]`                                            | `fc`                                         |
| 헤드 입력 차원 | 1792                                                       | 2048                                         |
| Optimizer 등록 | `model.parameters()`                                       | `model.parameters()`                         |
| 모델 저장명    | `best_model_efficiB4.pth`                                  | `best_model_resNet50.pth`                    |
| GradCAM target | `[model.features[-1]]`                                     | `[model.layer4[-1]]`                         |

> **[핵심 변경 요약] EfficientNetB4 → ResNet-50 전환 시 반드시 수정해야 할 3가지**
>
> 1. **Fine-tune 대상**
>    `model.features[-1]` → `model.layer4`
>    ResNet-50은 `features` 속성이 없고 `layer1~layer4`로 스테이지가 나뉜다.
>    마지막 Bottleneck 그룹인 `layer4`를 unfreeze하여 고수준 특징을 조정한다.
> 2. **분류 헤드 교체 위치 및 입력 차원**
>    `classifier[1] = nn.Linear(1792, 14)` → `fc = nn.Linear(2048, 14)`
>    ResNet-50은 Sequential 헤드가 없고 `fc` 단일 속성으로 분류 레이어가 붙는다.
>    출력 특징 차원도 1792 → 2048로 달라진다.
> 3. **GradCAM target layer**
>    `[model.features[-1]]` → `[model.layer4[-1]]`
>    ResNet-50에서 마지막 Conv 출력은 `layer4`의 마지막 Bottleneck 블록이다.
>    `.ipynb` 패턴: `target_layers = [model.layer4[-1]]`

---

## 구현 단계 (Phase)

### Phase 1 — 데이터 레이어

**1. Config 상수 정의**

- `NUM_CLASSES = 14`
- `EPOCHS = 20`, `OPTUNA_EPOCHS = 5`, `N_TRIALS = 20`
- `PATIENCE = 3`
- `BEST_MODEL_PATH = "best_model_resNet50.pth"`
- 이미지/CSV 경로: `../data2/`

**2. Transform 정의**

- `transform_train`: 224×224 리사이즈 + `ColorJitter(brightness=0.2, contrast=0.2)` + ImageNet 정규화
- `transform_val/test`: 리사이즈 + ImageNet 정규화만 적용 (증강 없음)

> **`RandomHorizontalFlip` 제외 근거**: 전입신고서는 정형화된 행정 양식 스캔 이미지이므로 좌우 반전 시 필드 위치가 역전되어 역효과 발생.
> `.ipynb`는 인물 이미지 기준이므로 `RandomHorizontalFlip`을 사용하나, 본 프로젝트에서는 `ColorJitter`만 사용한다.

**3. MultiLabelDataset (Custom Dataset)**

- `ImageFolder` 사용 불가 — 플랫 디렉토리이며 레이블이 CSV에 존재
- CSV `filename` 컬럼으로 이미지 로드
- 나머지 14컬럼 → `float32` 텐서로 반환

```python
# =====================================================================
# [주의] ImageFolder를 사용하지 않는 이유
# =====================================================================
# 는 ImageFolder(클래스별 서브폴더) 구조를 사용하지만,
# 이 프로젝트는 data2/train/ 아래 모든 이미지가 하나의 폴더에 평탄(flat)하게 존재하며
# 레이블은 별도 CSV에 14개 이진값으로 정의된다.
# → torch.utils.data.Dataset을 직접 상속한 커스텀 클래스를 사용한다.
# =====================================================================
```

---

### Phase 2 — 모델 구성

**4. `build_model()` 함수**

```python
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 전체 파라미터 freeze
for param in model.parameters():
    param.requires_grad = False

# layer4 (마지막 Bottleneck 그룹) unfreeze — Fine-Tuning 대상
# layer4 구성: Bottleneck × 3 (3개 블록, 각 블록: 1×1→3×3→1×1 Conv)
for param in model.layer4.parameters():
    param.requires_grad = True

# 분류 헤드 교체: fc의 Linear(2048, 1000) → Linear(2048, NUM_CLASSES)
model.fc = nn.Linear(2048, NUM_CLASSES)
```

> `.ipynb`의 Fine-Tuning 패턴 적용 (ResNet-34 기준 → ResNet-50으로 변경)
>
> - `layer1~layer3` : 저수준~중수준 범용 특징(엣지, 텍스처, 형태) → 동결 유지
> - `layer4` : 고수준 의미적 특징 담당 → 우리 도메인에 맞게 조정
> - `fc` : 분류 헤드 — 새로 초기화된 레이어이므로 항상 학습됨

---

### Phase 3 — Optuna 하이퍼파라미터 최적화

**5. `objective(trial)` 함수 정의**

- 탐색 파라미터:
  - `batch_size`: categorical `[4, 8, 16]`
  - `lr`: loguniform `1e-5 ~ 1e-3`
- 손실함수: `BCEWithLogitsLoss` (Multi-label)
- 옵티마이저: `optim.Adam(model.parameters(), lr=lr)`
- `OPTUNA_EPOCHS = 5` 루프로 빠른 탐색
- `trial.report(val_loss, epoch)` + `trial.should_prune()` 으로 조기 종료

**6. 최적화 실행**

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print(study.best_trial.params)
```

---

### Phase 4 — 최종 학습 & 평가

**7. 최적 파라미터로 재학습**

- Optuna에서 찾은 `batch_size`, `lr`로 DataLoader 재생성
- `tqdm` 진행 표시
- Early Stopping (patience = 3): val_loss 개선 시 `best_model_resNet50.pth` 저장
- **TensorBoard 기록** (`SummaryWriter` 사용)

**TensorBoard 기록 항목**

| 태그 | 기록 시점 | x축 기준 | 내용 |
| ------------- | --------- | ------------------------- | ------------------------------ |
| `Loss/train`  | 배치마다  | `global_step` (배치 누적) | 배치별 학습 손실 |
| `Loss/val`    | 에포크마다 | `epoch`                  | 에포크 평균 검증 손실 |
| `Acc/val`     | 에포크마다 | `epoch`                  | element-wise 검증 정확도 |

```python
writer = SummaryWriter()   # runs/<타임스탬프>/ 에 자동 저장
global_step = 0

# 배치 루프 내
writer.add_scalar("Loss/train", loss.item(), global_step)
global_step += 1

# 에포크 끝 — val loss/acc 동시 기록
writer.add_scalar("Loss/val", total_val_loss, epoch)
writer.add_scalar("Acc/val", val_acc, epoch)
writer.flush()   # 매 에포크마다 디스크에 즉시 기록

# 학습 완료 후
writer.close()
```

> **확인 방법**
> ```bash
> tensorboard --logdir=runs
> # → 브라우저에서 http://localhost:6006 접속 → Scalars 탭
> ```

```python
# =====================================================================
# [Early Stopping] patience = 3 의 동작 방식
# =====================================================================
# val_loss가 연속 3번 개선되지 않으면 학습을 조기 종료한다.
# 개선이 있을 때마다 best_model_resNet50.pth 를 덮어저장하며,
# 종료 시점이 아닌 "가장 좋았던 시점"의 모델이 최종 결과물이 된다.
#
# 의 early stopping 구조 적용
#   stop_count = 3 → PATIENCE = 3
#   early_stop_count 카운터로 연속 미개선 횟수 추적
# =====================================================================
```

**8. Test 평가**

- `sigmoid(output) > 0.5` → 이진 예측
- Element-wise Accuracy 출력

---

### Phase 5 — GradCAM 시각화

**9. GradCAM + GradCAMPlusPlus**

- `best_model_resNet50.pth` 로드
- test셋 첫 번째 이미지 자동 선택
- `target_layers = [model.layer4[-1]]` — `layer4`의 마지막 Bottleneck 블록
- `ClassifierOutputTarget(pred_class)` — 가장 높은 confidence 클래스 기준
- subplot 2개로 GradCAM / GradCAMPlusPlus 비교 시각화
- 결과 `gradcam_resNet50_result.png` 저장

> `.ipynb`의 GradCAM 코드 구조 적용
> `target_layers = [model.layer4[-1]]` 패턴 그대로 사용

---

## 참조 파일

| 파일                              | 참조 내용                                                                  |
| --------------------------------- | -------------------------------------------------------------------------- |
| `efficiNetB4/efficiNetB4_main.py` | 전체 파이프라인 구조 (Dataset, Optuna, Early Stopping, GradCAM)            |
| `resNet/.ipynb`     | ResNet layer4 unfreeze, fc 교체, Early Stopping, GradCAM target layer 패턴, TensorBoard `SummaryWriter` 패턴 |
| `resNet/modelselect.md`           | ResNet-50 선택 근거 및 데이터 분석                                         |
| `data2/train_labels.csv` 등       | 14개 이진 레이블, 파일명 참조                                              |

---

## 결정 사항 요약

| 항목                    | 결정값                                    | 근거                                  |
| ----------------------- | ----------------------------------------- | ------------------------------------- |
| 입력 해상도             | 224×224                                   | ResNet ImageNet 사전학습 기준         |
| Fine-tune 대상          | `layer4` (Bottleneck × 3)                 | 고수준 의미 특징, 도메인 적응 최적    |
| 분류 헤드               | `fc = Linear(2048, 14)`                   | ResNet-50 출력 차원 2048              |
| Optuna n_trials         | 20                                        | 탐색 효율과 시간의 균형               |
| Optuna epochs           | 5                                         | 탐색 시 빠른 수렴 판단                |
| 손실함수                | BCEWithLogitsLoss                         | Multi-label 이진 분류                 |
| Early Stopping patience | 3                                         | 소규모 데이터 + EPOCHS=20 조합에 적합 |
| 데이터 증강             | ColorJitter(brightness=0.2, contrast=0.2) | 행정 서식 도메인 특성 (반전 부적합)   |
| GradCAM target          | `[model.layer4[-1]]`                      | ResNet 마지막 Conv 출력               |
| 모델 저장명             | `best_model_resNet50.pth`                 | EfficientNetB4·ConvNeXt와 이름 구분   |
| TensorBoard 기록        | `Loss/train`(배치), `Loss/val`+`Acc/val`(에포크) | 학습 곡선·과적합 여부 실시간 모니터링 |
