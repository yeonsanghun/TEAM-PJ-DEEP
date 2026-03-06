import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# =====================================================================
# Config
# =====================================================================
NUM_CLASSES = 14
THRESHOLD = 0.5
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../efficiNetB4/best_model_efficiB4.pth")
)
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

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# =====================================================================
# 전역 모델 (서버 시작 시 1회 로드)
# =====================================================================
_model: nn.Module | None = None


def get_model() -> nn.Module:
    global _model
    if _model is None:
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(1792, NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


# =====================================================================
# 추론 + GradCAM++
# =====================================================================
def run_inference(image: Image.Image):
    model = get_model()

    # gradient 계산 활성화 (GradCAM 필요)
    for param in model.parameters():
        param.requires_grad = True

    input_tensor = transform(image).unsqueeze(0)
    pred = model(input_tensor)
    probs = torch.sigmoid(pred).squeeze().tolist()
    pred_class = int(torch.sigmoid(pred).argmax().item())

    # GradCAM++
    cam_pp = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])
    gradcam_map = cam_pp(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_class)],
    )[0]

    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    gradcam_overlay = show_cam_on_image(rgb_img, gradcam_map, use_rgb=True)

    # GradCAM 이미지 → base64 PNG
    pil_cam = Image.fromarray(gradcam_overlay)
    buf = io.BytesIO()
    pil_cam.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    predictions = [
        {
            "label": LABEL_COLS[i],
            "prob": float(probs[i]),
            "pred": bool(probs[i] >= THRESHOLD),
        }
        for i in range(NUM_CLASSES)
    ]

    return {
        "pred_class": pred_class,
        "pred_class_name": LABEL_COLS[pred_class],
        "predictions": predictions,
        "gradcam_b64": gradcam_b64,
        "n_filled": sum(1 for p in predictions if p["pred"]),
        "total": NUM_CLASSES,
    }


# =====================================================================
# FastAPI 앱
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # 서버 시작 시 모델 사전 로드
    yield


app = FastAPI(title="전입신고서 분석 API", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "전입신고서 필드 분석 API",
        "model": "EfficientNet-B4 Multi-Label",
        "labels": LABEL_COLS,
        "endpoints": {
            "GET /": "서비스 정보",
            "POST /efficiNetB4": "이미지 업로드 → 필드 예측 + GradCAM++ 반환",
        },
    }


@app.post("/efficiNetB4")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "이미지를 열 수 없습니다."}
        )

    result = run_inference(image)
    return JSONResponse(content=result)
