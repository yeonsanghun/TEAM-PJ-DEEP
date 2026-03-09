import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# 한글 폰트 설정 (Windows 맑은 고딕) — import 직후 최우선 적용
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# =====================================================================
# Config
# =====================================================================
NUM_CLASSES = 14
EPOCHS = 20
OPTUNA_EPOCHS = 5
N_TRIALS = 20
PATIENCE = 3
BEST_MODEL_PATH = "best_model_efficiB4.pth"

TRAIN_CSV = "../data2/train_labels.csv"
VAL_CSV = "../data2/val_labels.csv"
TEST_CSV = "../data2/test_labels.csv"
TRAIN_IMG_DIR = "../data2/train"
VAL_IMG_DIR = "../data2/val"
TEST_IMG_DIR = "../data2/test"

LABEL_COLS = [
    "전입자_성명",
    "전입자_주민등록번호",
    "전입자_연락처",
    "전입자_서명도장",
    "전_시도",
    "전_시군구",
    "현_세대주성명",
    "현_연락처",
    "현_서명도장",
    "현_주소",
    "전입사유_체크",
    "우편물서비스_동의체크",
    "신청인_성명",
    "신청인_서명도장",
]

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print('device:' + device)

# =====================================================================
# Phase 1 — Transform 정의
# =====================================================================
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# =====================================================================
# Phase 1 — MultiLabelDataset (Custom Dataset)
# =====================================================================
# [주의] ImageFolder를 사용하지 않는 이유
# -----------------------------------------------------------------------
# ImageFolder는 클래스별 서브폴더 구조(train/<클래스명>/이미지)를 전제로 동작한다.
# 이 프로젝트는 data2/train/ 아래 모든 이미지가 하나의 폴더에 평탄(flat)하게 존재하며,
# 레이블은 이미지 폴더가 아닌 별도 CSV에 14개 이진값으로 정의된다.
# → torch.utils.data.Dataset을 직접 상속한 커스텀 클래스를 사용한다.
# -----------------------------------------------------------------------
class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return img, label


# =====================================================================
# Phase 2 — 모델 구성
# =====================================================================
def build_model():
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    # 전체 파라미터 freeze
    for param in model.parameters():
        param.requires_grad = False

    # features[-1] (마지막 MBConv 블록)만 unfreeze → Fine-tuning 대상
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # 분류 헤드 교체: 1792 → NUM_CLASSES(14)
    model.classifier[1] = nn.Linear(1792, NUM_CLASSES)

    return model.to(device)


# =====================================================================
# Phase 3 — Optuna 하이퍼파라미터 최적화
# =====================================================================
train_dataset = MultiLabelDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform_train)
val_dataset = MultiLabelDataset(VAL_CSV, VAL_IMG_DIR, transform_val)


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = float("inf")

    for epoch in range(OPTUNA_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            preds = model(imgs.to(device))
            loss = criterion(preds, labels.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(device))
                val_loss += criterion(preds, labels.to(device)).item()

        total_loss = val_loss / len(val_loader)

        trial.report(total_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return total_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("=" * 50)
print("Best params:", study.best_trial.params)
print("=" * 50)

best_batch_size = study.best_trial.params["batch_size"]
best_lr = study.best_trial.params["lr"]


# =====================================================================
# Phase 4 — 최적 파라미터로 재학습
# =====================================================================
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

model = build_model()
optimizer = optim.Adam(model.parameters(), lr=best_lr)
criterion = nn.BCEWithLogitsLoss()

best_val_loss = float("inf")
early_stop_count = 0

# -----------------------------------------------------------------------
# [tqdm] 학습 진행 표시
# 사용 패턴: tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
#
# 정상 실행 시 터미널 출력 예시:
#   Epoch [1/20]: 100%|████████████| 81/81 [00:23<00:00, 3.45it/s, loss=0.423]
#   Epoch [2/20]: 100%|████████████| 81/81 [00:22<00:00, 3.62it/s, loss=0.391]
#
#   81/81         → 전체 배치 수 (train 샘플 649 ÷ batch_size)
#   [00:23<00:00] → 경과 시간 / 예상 남은 시간
#   3.45it/s      → 초당 처리 배치 수
#   loss=0.423    → 현재 배치 train loss (set_postfix로 표시)
# -----------------------------------------------------------------------
#
# [Early Stopping] patience = 3 의 동작 방식
# -----------------------------------------------------------------------
# val_loss가 연속 3번 개선되지 않으면 학습을 조기 종료한다.
# 개선이 있을 때마다 best_model_efficiB4.pth 를 덮어저장하며,
# 종료 시점이 아닌 "가장 좋았던 시점"의 모델이 최종 결과물이 된다.
#
# 예시 (EPOCHS=20 기준):
#   Epoch 1 : val_loss=0.45 → 갱신 ✓  (저장, count=0)
#   Epoch 2 : val_loss=0.42 → 갱신 ✓  (저장, count=0)
#   Epoch 3 : val_loss=0.44 → 개선 없음 (count=1)
#   Epoch 4 : val_loss=0.46 → 개선 없음 (count=2)
#   Epoch 5 : val_loss=0.45 → 개선 없음 (count=3) → 학습 중단!
#   → 최종 모델 = Epoch 2 시점의 가중치
#
# patience 값 선택 근거:
#   patience=1 : 너무 민감 — val_loss 소폭 상승 시 즉시 종료, 지역 최솟값 위험
#   patience=3 : 균형 OK  — 소규모 데이터(649장)+EPOCHS=20 조합에 적합
#                           자연적 loss 진동(노이즈) 1~2 epoch 허용 후 종료
#   patience=5+ : 느슨   — 과적합 구간을 오래 허용, best_model과 gap 커짐
# -----------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
    for imgs, labels in pbar:
        optimizer.zero_grad()
        preds = model(imgs.to(device))
        loss = criterion(preds, labels.to(device))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device))
            val_loss += criterion(preds, labels.to(device)).item()

    total_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch}/{EPOCHS}]  val_loss: {total_val_loss:.4f}")

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  → Model saved (best_val_loss: {best_val_loss:.4f})")
    else:
        early_stop_count += 1
        if early_stop_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break


# =====================================================================
# Phase 4 — Test 평가
# =====================================================================
test_dataset = MultiLabelDataset(TEST_CSV, TEST_IMG_DIR, transform_val)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs.to(device))
        pred_binary = (torch.sigmoid(preds) > 0.5).float()
        correct += (pred_binary == labels.to(device)).sum().item()
        total += labels.numel()

acc = correct / total
print(f"Test Element-wise Accuracy: {acc:.4f}")


# =====================================================================
# Phase 5 — GradCAM 시각화
# =====================================================================
def visualize_gradcam_pp(model, img_pil, filename, save_path="gradcam_result.png"):
    """GradCAM++ 히트맵을 생성하고 저장한다.

    Returns:
        dict:
            pred_class  (int)         - 가장 높은 확률의 클래스 인덱스
            probs       (list[float]) - 14개 클래스별 sigmoid 확률
            gradcam_img (np.ndarray)  - GradCAM++ 오버레이 이미지 (H×W×3, uint8)
    """
    for param in model.parameters():
        param.requires_grad = True

    input_tensor = transform_val(img_pil).unsqueeze(0).to(device)

    pred = model(input_tensor)
    probs = torch.sigmoid(pred).squeeze().tolist()
    pred_class = int(torch.sigmoid(pred).argmax().item())

    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(pred_class)]

    cam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
    gradcam_pp_map = cam_pp(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    vis_pp = show_cam_on_image(rgb_img, gradcam_pp_map, use_rgb=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(vis_pp)
    ax.set_title(f"GradCAM++ (class {pred_class}: {LABEL_COLS[pred_class]})")
    ax.axis("off")
    plt.suptitle(f"File: {filename}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved: {save_path}")

    return {
        "pred_class": pred_class,
        "probs": probs,
        "gradcam_img": vis_pp,
    }


model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# test셋 첫 번째 이미지 자동 선택 (재현성 보장)
first_row = test_dataset.df.iloc[0]
img_path = os.path.join(TEST_IMG_DIR, first_row["filename"])
img_pil = Image.open(img_path).convert("RGB")

result = visualize_gradcam_pp(model, img_pil, first_row["filename"])
print(f"pred_class : {result['pred_class']} ({LABEL_COLS[result['pred_class']]})")
print(f"probs      : {[f'{p:.4f}' for p in result['probs']]}")
print(
    f"gradcam_img: shape={result['gradcam_img'].shape}, dtype={result['gradcam_img'].dtype}"
)
