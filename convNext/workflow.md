# ConvNeXt-Small Multi-Label Classification — Workflow

## 개요

`data2/` 디렉토리의 전입신고서 이미지에 대해 14개 필드 기입 여부를 판별하는 **Multi-Label Classification** 파이프라인.
ConvNeXt-Small (pretrained `IMAGENET1K_V1`) 기반으로 Optuna 하이퍼파라미터 최적화와 GradCAM 시각화를 포함한다.

---

## 데이터 구조

| Split | 이미지 경로    | 레이블 파일              | 샘플 수 |
| ----- | -------------- | ------------------------ | ------- |
| Train | `data2/train/` | `data2/train_labels.csv` | 649     |
| Val   | `data2/val/`   | `data2/val_labels.csv`   | 147     |
| Test  | `data2/test/`  | `data2/test_labels.csv`  | 151     |

- **이미지 형식**: JPG, 플랫 디렉토리 구조 (클래스별 서브폴더 없음)
- **CSV 구조**: `filename` 컬럼 + 14개 이진(0/1) 레이블 컬럼
- **레이블 대상 (14개)**: 전입자*성명, 전입자*주민등록번호, 전입자*연락처, 전입자*서명도장, 전*시도, 전*시군구, 현*세대주성명, 현*연락처, 현*서명도장, 현*주소, 전입사유*체크, 우편물서비스*동의체크, 신청인*성명, 신청인*서명도장

---

## ConvNeXt-Small 아키텍처

```
features[0]  →  Stem (Patch Embedding)        224×224 → 56×56
features[1]  →  Downsampling + LayerNorm
features[2]  →  Stage 1 ConvNeXt Blocks
features[3]  →  Downsampling + LayerNorm
features[4]  →  Stage 2 ConvNeXt Blocks
features[5]  →  Downsampling + LayerNorm
features[6]  →  Stage 3 ConvNeXt Blocks       ← Fine-Tuning 대상
features[7]  →  LayerNorm                     ← Fine-Tuning 대상
avgpool      →  Global Average Pooling
classifier   →  [0] LayerNorm
                [1] Flatten
                [2] Linear(768, 14)            ← 교체 대상
```

- 파라미터 수: 약 **50M** (ImageNet Top-1 Acc. 83.1%)

---

## EfficientNetB4 vs ConvNeXt-Small 비교

| 항목           | EfficientNetB4                                             | ConvNeXt-Small                                                 |
| -------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| 모델 로드      | `efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)` | `convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)` |
| Fine-tune 대상 | `features[-1]` (마지막 MBConv 블록 1개)                    | `features[6]` + `features[7]` (Stage 3 + LayerNorm)            |
| 헤드 교체 위치 | `classifier[1]`                                            | `classifier[2]`                                                |
| 헤드 입력 차원 | 1792                                                       | 768                                                            |
| Optimizer 등록 | `model.parameters()`                                       | `filter(lambda p: p.requires_grad, model.parameters())`        |
| 모델 저장명    | `best_model_efficiB4.pth`                                  | `best_model_convNext.pth`                                      |
| GradCAM target | `[model.features[-1]]`                                     | `[model.features[-1]]` (= `features[7]`)                       |

> **[핵심 변경 요약] EfficientNetB4 → ConvNeXt-Small 전환 시 반드시 수정해야 할 3가지**
>
> 1. **Fine-tune 대상**
>    `features[-1]` (MBConv 블록 1개) → `features[6]` + `features[7]` (Stage 3 + LayerNorm) 2개 unfreeze
>    ConvNeXt는 마지막 스테이지가 `features[6]`이고, 그 뒤에 독립적인 LayerNorm(`features[7]`)이 분리되어 있어 함께 unfreeze해야 한다.
> 2. **분류 헤드 교체 위치 및 입력 차원**
>    `classifier[1]` 위치의 `Linear(1792, 14)` → `classifier[2]` 위치의 `Linear(768, 14)`
>    ConvNeXt의 classifier는 `[0] LayerNorm → [1] Flatten → [2] Linear` 구조이므로 인덱스가 다르고, 출력 특징 차원도 1792 → 768로 달라진다.
> 3. **Optimizer에 등록하는 파라미터 범위**
>    `model.parameters()` (전체) → `filter(lambda p: p.requires_grad, model.parameters())` (unfreeze된 것만)
>    ConvNeXtl는 freeze/unfreeze 대상이 2개(features[6], features[7])로 명시적으로 분리되므로, requires_grad=True인 파라미터만 옵티마이저에 전달하는 것이 안전하다.

---

## 구현 단계 (Phase)

### Phase 1 — 데이터 레이어

**1. Config 상수 정의**

- `NUM_CLASSES = 14`
- `EPOCHS = 20`, `OPTUNA_EPOCHS = 5`, `N_TRIALS = 20`
- `PATIENCE = 3`
- `BEST_MODEL_PATH = "best_model_convNext.pth"`
- 이미지/CSV 경로: `../data2/`

**2. Transform 정의**

- `transform_train`: 224×224 리사이즈 + `ColorJitter(brightness=0.2, contrast=0.2)` + ImageNet 정규화
- `transform_val/test`: 리사이즈 + ImageNet 정규화만 적용 (증강 없음)

> **`RandomHorizontalFlip` 제외 근거**: 전입신고서는 정형화된 행정 양식 스캔 이미지이므로 좌우 반전 시 필드 위치가 역전되어 역효과 발생. 스캐너·복사기 밝기/대비 편차에 대응하는 `ColorJitter`만 사용.

**3. MultiLabelDataset (Custom Dataset)**

- `ImageFolder` 사용 불가 — 플랫 디렉토리이며 레이블이 CSV에 존재
- CSV `filename` 컬럼으로 이미지 로드
- 나머지 14컬럼 → `float32` 텐서로 반환

```python
# =====================================================================
# [주의] ImageFolder를 사용하지 않는 이유
# =====================================================================
# ImageFolder는 클래스별 서브폴더 구조(train/<클래스명>/이미지)를 전제로 동작한다.
# 이 프로젝트는 data2/train/ 아래 모든 이미지가 하나의 폴더에 평탄(flat)하게 존재하며,
# 레이블은 이미지 폴더가 아닌 별도 CSV에 14개 이진값으로 정의된다.
# → torch.utils.data.Dataset을 직접 상속한 커스텀 클래스를 사용한다.
# =====================================================================
```

---

### Phase 2 — 모델 구성

**4. `build_model()` 함수**

```python
model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)

# 전체 파라미터 freeze
for param in model.parameters():
    param.requires_grad = False

# features[6] (Stage 3) + features[7] (최종 LayerNorm) unfreeze → Fine-Tuning 대상
for param in model.features[6].parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True

# 분류 헤드 교체: 768 → NUM_CLASSES(14)
model.classifier[2] = nn.Linear(768, NUM_CLASSES)
```

> `wd8_skin_fine_convNext.ipynb` STEP 13 Fine-Tuning 구성 패턴 적용
>
> - `features[0~5]` : 저수준 범용 특징(엣지, 텍스처) → 동결 유지
> - `features[6]` : 고수준 의미적 특징 담당 → 우리 도메인에 맞게 조정
> - `features[7]` : 최종 LayerNorm → 특징 정규화 스케일 조정

---

### Phase 3 — Optuna 하이퍼파라미터 최적화

**5. `objective(trial)` 함수 정의**

- 탐색 파라미터:
  - `batch_size`: categorical `[4, 8, 16]`
  - `lr`: loguniform `1e-5 ~ 1e-3`
- 손실함수: `BCEWithLogitsLoss` (Multi-label)
- 옵티마이저: `filter(lambda p: p.requires_grad, model.parameters())` 로 unfreeze 파라미터만 등록
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
- Early Stopping (patience = 3): val_loss 개선 시 `best_model_convNext.pth` 저장
- **TensorBoard 기록** (`SummaryWriter` 사용)

**TensorBoard 기록 항목**

| 태그 | 기록 시점 | x축 기준 | 참조 노트북 대응 |
| ------------- | --------- | ------------------------- | --------------------------------- |
| `Loss/train`  | 배치마다  | `global_step` (배치 누적) | `writer.add_scalar("Loss/train")` |
| `Loss/val`    | 에포크마다 | `epoch`                  | `writer_ft.add_scalar(...)` |
| `Acc/val`     | 에포크마다 | `epoch`                  | element-wise accuracy |

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
# 개선이 있을 때마다 best_model_convNext.pth 를 덮어저장하며,
# 종료 시점이 아닌 "가장 좋았던 시점"의 모델이 최종 결과물이 된다.
#
# 예시 (EPOCHS=20 기준):
#   Epoch 1 : val_loss=0.45 → 갱신 ✅  (저장, count=0)
#   Epoch 2 : val_loss=0.42 → 갱신 ✅  (저장, count=0)
#   Epoch 3 : val_loss=0.44 → 개선 없음 (count=1)
#   Epoch 4 : val_loss=0.46 → 개선 없음 (count=2)
#   Epoch 5 : val_loss=0.45 → 개선 없음 (count=3) → 학습 중단!
#   → 최종 모델 = Epoch 2 시점의 가중치
# =====================================================================
```

**8. Test 평가**

- `sigmoid(output) > 0.5` → 이진 예측
- Element-wise Accuracy 출력

---

### Phase 5 — GradCAM 시각화

**9. GradCAM + GradCAMPlusPlus**

- `best_model_convNext.pth` 로드
- test셋 첫 번째 이미지 자동 선택
- `target_layers = [model.features[-1]]` (= `features[7]`, 최종 LayerNorm)
- `ClassifierOutputTarget(pred_class)` — 가장 높은 confidence 클래스 기준
- subplot 2개로 GradCAM / GradCAMPlusPlus 비교 시각화
- 결과 `gradcam_convNext_result.png` 저장

> `wd8_skin_fine_convNext.ipynb` STEP 14 GradCAM 구조 적용

---

## 참조 파일

| 파일                                    | 참조 내용                                                                |
| --------------------------------------- | ------------------------------------------------------------------------ |
| `efficiNetB4/efficiNetB4_main.py`       | 전체 파이프라인 구조 (Dataset, Optuna, Early Stopping, GradCAM)          |
| `convNext/wd8_skin_fine_convNext.ipynb` | ConvNeXt 모델 로드·freeze·unfreeze·헤드교체·Fine-Tuning 패턴 (STEP 9~14), TensorBoard `SummaryWriter` 패턴 (Cell 31, 37) |
| `data2/train_labels.csv` 등             | 14개 이진 레이블, 파일명 참조                                            |

---

## 결정 사항 요약

| 항목                    | 결정값                                    | 근거                                       |
| ----------------------- | ----------------------------------------- | ------------------------------------------ |
| 입력 해상도             | 224×224                                   | ConvNeXt ImageNet 사전학습 기준            |
| Fine-tune 대상          | `features[6]` + `features[7]`             | 고수준 의미 특징 담당, 도메인 적응 최적    |
| 분류 헤드               | `classifier[2]` = `Linear(768, 14)`       | ConvNeXt-Small 출력 차원 768               |
| Optuna n_trials         | 20                                        | 탐색 효율과 시간의 균형                    |
| Optuna epochs           | 5                                         | 탐색 시 빠른 수렴 판단                     |
| 손실함수                | BCEWithLogitsLoss                         | Multi-label 이진 분류                      |
| Early Stopping patience | 3                                         | 소규모 데이터(649장)+EPOCHS=20 조합에 적합 |
| 데이터 증강             | ColorJitter(brightness=0.2, contrast=0.2) | 행정 서식 도메인 특성 (반전 부적합)        |
| GradCAM target          | test셋 첫 번째 이미지 (자동)              | 재현성 보장                                |
| 모델 저장명             | `best_model_convNext.pth`                 | EfficientNetB4와 이름 구분                 |
| TensorBoard 기록        | `Loss/train`(배치), `Loss/val`+`Acc/val`(에포크) | 학습 곡선·과적합 여부 실시간 모니터링 |
