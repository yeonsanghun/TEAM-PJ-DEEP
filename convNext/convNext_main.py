import os

# 한글 폰트 설정 (Linux NanumGothic) — import 직후 최우선 적용
import matplotlib.font_manager as _fm
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
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm

_fm._load_fontmanager(try_read_cache=False)  # 폰트 캐시 강제 갱신
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# =====================================================================
# Config
# =====================================================================
NUM_CLASSES = 14
EPOCHS = 20
OPTUNA_EPOCHS = 5
N_TRIALS = 20
PATIENCE = 3
BEST_MODEL_PATH = "best_model_convNext.pth"

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
# [전입신고서 도메인 특성]
# 정형화된 행정 양식 스캔 이미지이므로 RandomHorizontalFlip 제외.
#   - 좌우 반전 시 필드 위치가 역전 → 역효과
# ColorJitter만 사용하여 스캐너·복사기 밝기/대비 편차에 대응.
transform_train = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_val = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
# [EfficientNetB4 → ConvNeXt-Small 핵심 변경 3가지]
#
# 1. Fine-tune 대상
#    EfficientNetB4 : features[-1] (마지막 MBConv 블록 1개)
#    ConvNeXt-Small : features[6] (Stage 3) + features[7] (최종 LayerNorm) 2개
#    → ConvNeXt는 마지막 스테이지(features[6]) 뒤에 독립적인 LayerNorm(features[7])이
#      분리되어 있어 함께 unfreeze해야 한다.
#
# 2. 분류 헤드 교체 위치 및 입력 차원
#    EfficientNetB4 : classifier[1] = Linear(1792, 14)
#    ConvNeXt-Small : classifier[2] = Linear(768, 14)
#    → ConvNeXt classifier 구조: [0]LayerNorm → [1]Flatten → [2]Linear
#      인덱스가 다르고, 출력 특징 차원도 1792 → 768로 달라진다.
#
# 3. Optimizer에 등록하는 파라미터 범위
#    EfficientNetB4 : optim.Adam(model.parameters(), lr=lr)
#    ConvNeXt-Small : optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#    → freeze/unfreeze 대상이 2개로 명시적으로 분리되므로
#      requires_grad=True인 파라미터만 옵티마이저에 전달하는 것이 안전하다.
# =====================================================================
def build_model():
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)

    # 전체 파라미터 freeze
    for param in model.parameters():
        param.requires_grad = False

    # features[6] (Stage 3 ConvNeXt Blocks) unfreeze — 고수준 의미 특징 조정
    for param in model.features[6].parameters():
        param.requires_grad = True

    # features[7] (최종 LayerNorm) unfreeze — 특징 정규화 스케일 조정
    for param in model.features[7].parameters():
        param.requires_grad = True

    # 분류 헤드 교체: classifier[2]의 Linear(768, 1000) → Linear(768, NUM_CLASSES)
    model.classifier[2] = nn.Linear(768, NUM_CLASSES)

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

    # unfreeze된 파라미터(features[6], features[7], classifier[2])만 옵티마이저에 등록
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=best_lr
)
criterion = nn.BCEWithLogitsLoss()

best_val_loss = float("inf")
early_stop_count = 0
writer = SummaryWriter()
global_step = 0

# -----------------------------------------------------------------------
# [Early Stopping] patience = 3 의 동작 방식
# -----------------------------------------------------------------------
# val_loss가 연속 3번 개선되지 않으면 학습을 조기 종료한다.
# 개선이 있을 때마다 best_model_convNext.pth 를 덮어저장하며,
# 종료 시점이 아닌 "가장 좋았던 시점"의 모델이 최종 결과물이 된다.
#
# 예시 (EPOCHS=20 기준):
#   Epoch 1 : val_loss=0.45 → 갱신 ✓  (저장, count=0)
#   Epoch 2 : val_loss=0.42 → 갱신 ✓  (저장, count=0)
#   Epoch 3 : val_loss=0.44 → 개선 없음 (count=1)
#   Epoch 4 : val_loss=0.46 → 개선 없음 (count=2)
#   Epoch 5 : val_loss=0.45 → 개선 없음 (count=3) → 학습 중단!
#   → 최종 모델 = Epoch 2 시점의 가중치
# -----------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
    for imgs, labels in pbar:
        optimizer.zero_grad()
        preds = model(imgs.to(device))
        loss = criterion(preds, labels.to(device))
        # TensorBoard — 배치별 학습 손실 기록
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device))
            val_loss += criterion(preds, labels.to(device)).item()
            pred_binary = (torch.sigmoid(preds) > 0.5).float()
            val_correct += (pred_binary == labels.to(device)).sum().item()
            val_total += labels.numel()

    total_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    # TensorBoard — 에포크별 검증 손실 및 정확도 기록
    writer.add_scalar("Loss/val", total_val_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
    writer.flush()
    print(f"Epoch [{epoch}/{EPOCHS}]  val_loss: {total_val_loss:.4f}  val_acc: {val_acc:.4f}")

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

writer.close()


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
# target_layers = [model.features[-1]]
#   ConvNeXt-Small에서 features[-1]은 features[7] (최종 LayerNorm)에 해당한다.
#   EfficientNetB4의 features[-1]과 표기는 동일하나 실제 레이어 종류가 다르므로 주의.
def visualize_gradcam_pp(
    model, img_pil, filename, save_path="gradcam_convNext_result.png"
):
    """GradCAM++ 히트맵을 생성하고 저장한다.

    Returns:
        dict:
            pred_class  (int)         - 가장 높은 확률의 클래스 인덱스
            probs       (list[float]) - 14개 클래스별 sigmoid 확률
            gradcam_img (np.ndarray)  - GradCAM++ 오버레이 이미지 (H×W×3, uint8)
    """
    # GradCAM 계산을 위해 모든 파라미터 기울기 활성화
    for param in model.parameters():
        param.requires_grad = True

    input_tensor = transform_val(img_pil).unsqueeze(0).to(device)

    pred = model(input_tensor)
    probs = torch.sigmoid(pred).squeeze().tolist()
    pred_class = int(torch.sigmoid(pred).argmax().item())

    # features[-1] = features[7] : ConvNeXt-Small 최종 LayerNorm
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
