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
BEST_MODEL_PATH = "best_model_resNet50.pth"

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device:' + device)
# =====================================================================
# Phase 1 — Transform 정의
# =====================================================================
# [전입신고서 도메인 특성]
# wd8_fine_final.ipynb 는 인물 이미지 기준으로 RandomHorizontalFlip 을 사용하지만,
# 전입신고서는 정형화된 행정 양식 스캔 이미지이므로 좌우 반전 시 필드 위치가 역전 → 역효과.
# 스캐너·복사기 밝기/대비 편차에 대응하는 ColorJitter 만 사용한다.
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
# wd8_fine_final.ipynb 는 ImageFolder(클래스별 서브폴더) 구조를 사용하지만,
# 이 프로젝트는 data2/train/ 아래 모든 이미지가 하나의 폴더에 평탄(flat)하게 존재하며
# 레이블은 별도 CSV에 14개 이진값으로 정의된다.
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
# [EfficientNetB4 → ResNet-50 핵심 변경 3가지]
#
# 1. Fine-tune 대상
#    EfficientNetB4 : model.features[-1]  (마지막 MBConv 블록)
#    ResNet-50      : model.layer4        (마지막 Bottleneck 그룹, Bottleneck × 3)
#    → ResNet-50은 features 속성이 없고 layer1~layer4로 스테이지가 나뉜다.
#      layer4를 unfreeze하여 고수준 의미 특징을 조정한다.
#
# 2. 분류 헤드 교체 위치 및 입력 차원
#    EfficientNetB4 : classifier[1] = nn.Linear(1792, 14)
#    ResNet-50      : fc             = nn.Linear(2048, 14)
#    → ResNet-50은 Sequential 헤드 없이 fc 단일 속성으로 분류 레이어가 붙는다.
#      출력 특징 차원도 1792 → 2048로 달라진다.
#
# 3. GradCAM target layer
#    EfficientNetB4 : [model.features[-1]]
#    ResNet-50      : [model.layer4[-1]]
#    → wd8_fine_final.ipynb 패턴: target_layers = [model.layer4[-1]]
#      layer4의 마지막 Bottleneck 블록이 GradCAM 대상이다.
# =====================================================================
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 전체 파라미터 freeze
    for param in model.parameters():
        param.requires_grad = False

    # layer4 (마지막 Bottleneck 그룹) unfreeze — Fine-Tuning 대상
    # layer4 구성: Bottleneck × 3 (각 블록: 1×1 → 3×3 → 1×1 Conv)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 분류 헤드 교체: fc의 Linear(2048, 1000) → Linear(2048, NUM_CLASSES)
    # 새로 생성된 Linear 레이어는 requires_grad=True(기본값)
    model.fc = nn.Linear(2048, NUM_CLASSES)

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
writer = SummaryWriter()
global_step = 0

# -----------------------------------------------------------------------
# [Early Stopping] patience = 3 의 동작 방식
# -----------------------------------------------------------------------
# val_loss가 연속 3번 개선되지 않으면 학습을 조기 종료한다.
# 개선이 있을 때마다 best_model_resNet50.pth 를 덮어저장하며,
# 종료 시점이 아닌 "가장 좋았던 시점"의 모델이 최종 결과물이 된다.
#
# wd8_fine_final.ipynb 의 early stopping 구조 동일 적용
#   stop_count = 3 → PATIENCE = 3
#   early_stop_count 카운터로 연속 미개선 횟수 추적
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
# target_layers = [model.layer4[-1]]
#   wd8_fine_final.ipynb 패턴 그대로 적용.
#   layer4의 마지막 Bottleneck 블록이 ResNet-50의 마지막 Conv 출력에 해당한다.
def visualize_gradcam_pp(
    model, img_pil, filename, save_path="gradcam_resNet50_result.png"
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

    # layer4[-1]: layer4의 마지막 Bottleneck 블록 — ResNet-50 최종 Conv 출력
    target_layers = [model.layer4[-1]]
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
