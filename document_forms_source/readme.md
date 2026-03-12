# 프로젝트 보고서: ConvNeXt Small 기반 서식 문서 다중 라벨 분류 시스템

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 행정 서식(PDF) 데이터를 기반으로 특정 항목의 기재 여부나 오류를 자동으로 감지하는 **딥러닝 기반 다중 라벨 분류(Multi-label Classification) 시스템** 구축을 목표로 합니다. 최신 CNN 아키텍처인 **ConvNeXt Small**을 활용하여 데이터 전처리부터 모델 최적화, Grad-CAM을 통한 시각적 분석까지 전 과정을 수행합니다.

## 2. 주요 워크플로우 (Workflow)
프로젝트는 총 4단계의 체계적인 과정을 거쳐 진행됩니다.
*   **Step 1: 데이터 전처리**: PDF에서 분석 대상인 첫 페이지만 추출한 뒤, 300 DPI의 고해상도 JPG 이미지로 변환하고 라벨 규칙에 따라 폴더를 구조화합니다.
*   **Step 2: 데이터셋 구축**: 데이터를 Train / Validation / Test 세트로 물리적으로 분리하고, 학습에 필요한 `labels.csv` 파일을 생성합니다.
*   **Step 3: 모델 학습 및 최적화**: ConvNeXt Small 백본에 14개 클래스 분류 헤드를 결합하고, **Optuna**를 통해 최적의 학습률과 배치 사이즈를 탐색합니다.
*   **Step 4: 성능 평가 및 분석**: 학습된 모델로 추론을 수행하고, **Grad-CAM** 히트맵을 통해 모델이 문서의 어느 영역을 보고 판단했는지 시각화합니다.

## 3. 모델 설계 및 학습 전략
### 3.1 왜 ConvNeXt Small인가?
*   **특징 추출**: 텍스트 영역, 표, 필드 위치 등 문서 이미지의 시각적 패턴을 정밀하게 포착합니다.
*   **효율성**: Transformer의 설계 철학을 CNN에 접목하여 높은 정확도와 GPU 메모리 효율(빠른 추론)을 동시에 달성했습니다.
*   **학습 용이성**: 구조적 패턴 학습에 강하며, ImageNet 사전학습 가중치를 통한 전이학습이 유리합니다.

### 3.2 모델 구성 및 하이퍼파라미터
*   **Architecture**: ConvNeXt Small 기반, 출력층 Linear(768 → 14).
*   **Loss Function**: 다중 라벨 분류에 적합한 **BCEWithLogitsLoss** 사용.
*   **주요 설정**: Batch Size 8, Optimizer Adam, 기본 Learning Rate 1e-4.
*   **데이터 증강**: Resize(224x224), Random Rotation, Affine, Perspective, Color Jitter 적용.
*   **Normalization**: 배경이 밝고 분산이 작은 문서 이미지 특성을 반영한 **커스텀 값**(Mean: [0.9367, 0.9364, 0.9358])을 적용하여 수렴 효율을 높였습니다.

## 4. 평가 지표: AP 및 mAP 상세 설명
모델의 성능을 다각도로 평가하기 위해 Accuracy, F1 Score와 함께 **mAP**를 핵심 지표로 활용합니다.

*   **AP (Average Precision)**: 특정 필드(항목) 하나에 대해 모델이 예측한 결과의 **정밀도(Precision)와 재현율(Recall) 사이의 관계**를 나타내는 곡선 아래 면적입니다. 이는 해당 항목을 얼마나 정확하고 빠짐없이 찾아내는지를 나타내는 단일 클래스 평가 척도입니다.
*   **mAP (mean Average Precision)**: 프로젝트에서 정의한 **14개 모든 필드에 대한 AP 값을 계산하여 평균을 낸 지표**입니다. 모델의 전체적인 다중 라벨 분류 성능을 종합적으로 보여줍니다.

## 5. 단계별 실험 결과 분석
*   **1회차 실험**: ImageNet 기본 설정을 적용했습니다. 정확도 0.9598, **mAP 0.9870**을 기록하며 우수한 초기 성능을 보였습니다.
*   **2회차 실험**: 손글씨 데이터를 포함하고 데이터셋을 4,000장으로 확장했습니다. **mAP는 0.9988**로 상승하며 변별력이 개선되었으나, 시각화 결과 실제 미기입 필드 외의 항목까지 오류로 판단하는 등 다소 불안정한 예측 양상이 관찰되었습니다.
*   **3회차 실험 (최종)**: **Optuna**를 활용해 배치 사이즈 8, 학습률 약 0.000737 부근에서 최적의 성능을 찾았습니다. **mAP 0.9998**이라는 수치를 달성했으며, 이전 실험 대비 정확도가 크게 향상되었습니다. 다만, 예측의 안정성을 더 높여야 할 부분은 여전히 과제로 남아 있습니다.

## 6. 주요 실행 명령어 (Execution Commands)
프로젝트 구동 및 학습 모니터링을 위해 다음 명령어를 사용합니다.

*   **스트림릿 모델 테스터 구동**:
    ```bash
    uv run streamlit run streamlit/model_tester_convnext.py
    ```
*   **텐서보드 실행 (학습 로그 확인)**:
    ```bash
    tensorboard --logdir=document_forms_source/runs
    ```

## 7. 환경 설정 및 주요 파일 (Environment & Files)
*   **언어 및 프레임워크**: Python 3.12, PyTorch 2.10, Torchvision 2.10, Optuna, Streamlit>=1.32, Pandas<3.
*   **핵심 스크립트**:
    *   `pdf_to_jpg.ipynb`: 300 DPI 고해상도 이미지 변환.
    *   `create_models_data2-optuna.ipynb`: 하이퍼파라미터 자동 튜닝.
    *   `load_models_data2.ipynb`: 최종 모델 로드 및 Grad-CAM 시각화 분석.

## 8. 주의사항 (Precautions)
*   **DPI 설정**: 미세한 텍스트 식별을 위해 PDF 변환 시 반드시 **300 DPI**를 유지해야 합니다.
*   **실행 환경**: 원활한 학습과 추론을 위해 **CUDA 가속이 가능한 GPU 환경** 사용을 권장합니다.
*   **경로 관리**: 코드 실행 전 각 노트북 상단의 `root_dir` 또는 `MODEL_PATH` 변수를 사용자 환경에 맞게 수정해야 합니다.