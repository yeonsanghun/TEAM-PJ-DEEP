# ============================================================
# 멀티레이블 분류 - CNN (VGG16)
# Notion 참고: 멀티레이블분류-CNN
#
# 전체 흐름:
#   1. 정답 데이터셋 형태 (image_path + 라벨 벡터 CSV)
#   2. 폴더 구조: dataset/images/ + labels.csv
#   3. PyTorch Dataset 정의 (MultiLabelDataset)
#   4. VGG16 멀티레이블 모델 (마지막 layer 수정)
#   5. Loss 함수: BCEWithLogitsLoss
#   6. 학습 코드
#   7. 추론 코드 (sigmoid + threshold)
#   8. 클래스 이름 매핑
# ============================================================

# ============================================================
# ## 라이브러리 임포트
# ============================================================
import csv
import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ## 3. PyTorch Dataset 코드
#
# CSV 형태:
#   image_path | A | B | C | D | E | F | G
#   img1.jpg   | 1 | 0 | 1 | 0 | 0 | 1 | 0
#
# - self.data.iloc[:, 0]   → image_path 열
# - self.data.iloc[:, 1:]  → 라벨 벡터 (binary, float32)
# ============================================================
class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 1열(image_path) 이후 모든 열이 라벨 벡터
        self.labels = self.data.iloc[:, 1:].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# ## 4. VGG16 멀티레이블 모델
#
# VGG16 마지막 layer(classifier[6])만 수정합니다.
#   기본: nn.Linear(4096, 1000)  → ImageNet 1000 클래스
#   수정: nn.Linear(4096, num_classes)  → 우리 클래스 수
# ============================================================
def build_model(num_classes: int, device: str) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # 마지막 FC 레이어 교체
    model.classifier[6] = nn.Linear(4096, num_classes)

    model = model.to(device)
    return model


# ============================================================
# ## 이미지 전처리 파이프라인
#
# VGG16은 ImageNet으로 사전 학습 → 동일한 mean/std로 정규화
# ============================================================
def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# ## 6. 학습 코드
#
# - Loss: BCEWithLogitsLoss  (멀티레이블 필수)
# - Optimizer: Adam
# - 에폭마다 평균 Loss 출력
# - 학습 완료 후 모델 가중치 저장
# ============================================================
def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int = 10,
    lr: float = 1e-4,
    save_path: str = "multilabel_vgg16.pth",
) -> None:
    # 5. Loss 함수 — 멀티레이블은 반드시 BCEWithLogitsLoss 사용
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}  Loss {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n모델 저장 완료 → {save_path}")


# ============================================================
# ## 7. 추론 코드
#
# 멀티레이블은 sigmoid 후 threshold(0.5) 사용합니다.
#   - softmax(X): 확률 합이 1 → 단일 클래스 선택
#   - sigmoid(O): 각 클래스 독립 확률 → 여러 클래스 동시 선택 가능
#
# 예시 결과: tensor([[1,0,1,0,0,1,0]])
# ============================================================
def infer_single(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        # (C,H,W) → (1,C,H,W) 배치 차원 추가
        output = model(image_tensor.unsqueeze(0).to(device))

        probs = torch.sigmoid(output)

        preds = (probs > threshold).int()

    return preds


# ============================================================
# ## 8. 클래스 이름 매핑
#
# 예측 벡터 [1,0,1,0,0,1,0] → ['A','C','F']
# ============================================================
def decode_labels(
    preds: torch.Tensor,
    classes: list[str],
) -> list[str]:
    pred_labels = [classes[i] for i, v in enumerate(preds[0]) if v == 1]
    return pred_labels


# ============================================================
# ## 더미 데이터 생성 헬퍼 (실제 이미지 데이터가 없을 때 테스트용)
#
# - dataset/images/ 폴더에 임의 색상의 PNG 이미지 생성
# - dataset/labels.csv 에 랜덤 이진 라벨 CSV 생성
# ============================================================
def create_dummy_dataset(
    dataset_dir: str,
    num_images: int = 30,
    num_classes: int = 7,
    class_names: list[str] | None = None,
    img_size: tuple[int, int] = (64, 64),
) -> tuple[str, str]:
    if class_names is None:
        class_names = [chr(ord("A") + i) for i in range(num_classes)]

    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    csv_path = os.path.join(dataset_dir, "labels.csv")

    rows = []
    for i in range(num_images):
        fname = f"img{i + 1:03d}.jpg"
        fpath = os.path.join(images_dir, fname)

        # 임의 색상 이미지 생성
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        img = Image.new("RGB", img_size, color=(r, g, b))
        img.save(fpath)

        # 랜덤 이진 라벨 (최소 1개 이상 1)
        while True:
            label_vec = [random.randint(0, 1) for _ in range(num_classes)]
            if sum(label_vec) > 0:
                break

        rows.append([fname] + label_vec)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + class_names)
        writer.writerows(rows)

    print(f"더미 데이터 생성 완료: {num_images}개 이미지, 라벨 {num_classes}개")
    print(f"  이미지: {images_dir}")
    print(f"  CSV  : {csv_path}")

    return images_dir, csv_path


# ============================================================
# ## 메인 실행
# ============================================================
def main():
    # ── 설정 ────────────────────────────────────────────────
    NUM_CLASSES = 7
    CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G"]
    EPOCHS = 3  # 테스트용 (실제 학습 시 10 이상 권장)
    BATCH_SIZE = 8
    LR = 1e-4
    THRESHOLD = 0.5

    # 현재 파일 기준 dataset 폴더
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")
    IMG_DIR = os.path.join(DATASET_DIR, "images")
    SAVE_PATH = os.path.join(BASE_DIR, "multilabel_vgg16.pth")

    # ── 디바이스 설정 ────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")

    # ── 더미 데이터 생성 (CSV 또는 images 폴더 없을 때만) ────
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMG_DIR):
        print("\n실제 데이터셋 없음 → 더미 데이터 자동 생성")
        IMG_DIR, CSV_PATH = create_dummy_dataset(
            dataset_dir=DATASET_DIR,
            num_images=40,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
        )

    # ── 이미지 전처리 ────────────────────────────────────────
    transform = get_transform()

    # ── Dataset & DataLoader ─────────────────────────────────
    dataset = MultiLabelDataset(
        csv_file=CSV_PATH,
        img_dir=IMG_DIR,
        transform=transform,
    )
    print(f"\n데이터셋 크기: {len(dataset)}개")

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ── 모델 빌드 ────────────────────────────────────────────
    # 출력 = NUM_CLASSES (7)
    model = build_model(num_classes=NUM_CLASSES, device=device)
    print(f"\nVGG16 마지막 레이어: {model.classifier[6]}")

    # ── 학습 ─────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"학습 시작  (epochs={EPOCHS}, lr={LR})")
    print(f"{'=' * 40}")
    train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        save_path=SAVE_PATH,
    )

    # ── 추론 예시 ────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print("추론 예시 (첫 번째 이미지)")
    print(f"{'=' * 40}")

    # 데이터셋의 첫 번째 샘플로 추론
    sample_image, sample_label = dataset[0]

    preds = infer_single(
        model=model,
        image_tensor=sample_image,
        device=device,
        threshold=THRESHOLD,
    )
    print(f"예측 벡터: {preds}")

    # 8. 클래스 이름 매핑
    pred_labels = decode_labels(preds, CLASS_NAMES)
    print(f"예측 라벨: {pred_labels}")

    # 정답과 비교
    gold_labels = [CLASS_NAMES[i] for i, v in enumerate(sample_label) if v == 1]
    print(f"정답 라벨: {gold_labels}")


if __name__ == "__main__":
    main()
