# ============================================================
# 멀티레이블 분류 FastAPI 추론 서버
# - VGG16 모델 (multi_label_main.py 에서 학습된 가중치 사용)
# - POST /infer  : 이미지 업로드 → 멀티레이블 예측 반환
# - GET  /       : 서버 상태 확인
# - GET  /classes: 클래스 목록 확인
#
# 실행 방법:
#   uvicorn multi_lable_fastapi:app --reload --port 8000
#   또는
#   python multi_lable_fastapi.py
#
# 테스트 (Swagger UI):
#   http://localhost:8000/docs
# ============================================================

import io
import os
import uuid
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ============================================================
# ## 설정
# ============================================================
# 클래스 이름 — multi_label_main.py 의 CLASS_NAMES 와 동일하게 맞출 것
CLASS_NAMES: list[str] = ["A", "B", "C", "D", "E", "F", "G"]
NUM_CLASSES: int = len(CLASS_NAMES)

# 모델 가중치 경로 (multi_label_main.py 실행 후 생성된 .pth)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multilabel_vgg16.pth")

# 업로드 이미지 저장 폴더
UPLOAD_DIR = os.path.join(BASE_DIR, "upload_img")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 추론 임계값 (sigmoid 확률 > THRESHOLD 이면 해당 클래스 예측)
THRESHOLD: float = 0.5

# 허용 확장자
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

# 디바이스
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ## 이미지 전처리 파이프라인
# multi_label_main.py 의 get_transform() 과 동일
# ============================================================
transform = transforms.Compose(
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
# ## 모델 로드 헬퍼
# ============================================================
def load_model(model_path: str, num_classes: int, device: str) -> nn.Module:
    model = models.vgg16(weights=None)  # 구조만 생성
    model.classifier[6] = nn.Linear(4096, num_classes)  # 마지막 레이어 교체

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            "먼저 multi_label_main.py 를 실행해 학습 및 저장하세요."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ============================================================
# ## FastAPI 생명주기 — 서버 시작 시 모델 로드
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Startup] 디바이스: {device}")
    print(f"[Startup] 모델 로드 중: {MODEL_PATH}")
    try:
        app.state.model = load_model(MODEL_PATH, NUM_CLASSES, device)
        print("[Startup] 모델 로드 완료!")
    except FileNotFoundError as e:
        print(f"[Startup] 경고: {e}")
        app.state.model = None
    yield
    print("[Shutdown] 서버 종료.")


# ============================================================
# ## FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="멀티레이블 분류 API",
    description=(
        "VGG16 기반 멀티레이블 이미지 분류 서버\n\n"
        "- **POST /infer** : 이미지 업로드 → 예측 라벨 반환\n"
        "- **GET  /classes**: 클래스 목록 확인\n\n"
        "먼저 `multi_label_main.py` 를 실행해 `multilabel_vgg16.pth` 를 생성하세요."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ## 유틸 함수
# ============================================================
def validate_and_save(img_bytes: bytes, filename: str) -> str:
    """확장자 검증 후 upload_img 폴더에 저장, 저장된 파일명 반환."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"허용되지 않는 파일 형식입니다. 허용: {ALLOWED_EXT}",
        )
    new_name = f"{uuid.uuid4()}.{ext}"
    with open(os.path.join(UPLOAD_DIR, new_name), "wb") as f:
        f.write(img_bytes)
    return new_name


def bytes_to_tensor(img_bytes: bytes) -> torch.Tensor:
    """bytes → PIL → 전처리 → (1, 3, 224, 224) 텐서."""
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, C, H, W)
    return tensor


def run_inference(model: nn.Module, tensor: torch.Tensor) -> dict:
    """텐서 → sigmoid → threshold → 예측 결과 딕셔너리."""
    with torch.no_grad():
        output = model(tensor)  # (1, num_classes) logits
        probs = torch.sigmoid(output)[0]  # (num_classes,) 확률
        preds = (probs > THRESHOLD).int()  # (num_classes,) 0/1

    pred_labels = [CLASS_NAMES[i] for i, v in enumerate(preds) if v == 1]
    scores = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    return {
        "predicted_labels": pred_labels,  # 예: ["A", "C", "F"]
        "predicted_vector": preds.tolist(),  # 예: [1,0,1,0,0,1,0]
        "scores": scores,  # 각 클래스별 sigmoid 확률
        "threshold": THRESHOLD,
    }


# ============================================================
# ## 엔드포인트 (1) — GET /  헬스체크
# ============================================================
@app.get("/", summary="서버 상태 확인")
def root():
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }


# ============================================================
# ## 엔드포인트 (2) — GET /classes  클래스 목록
# ============================================================
@app.get("/classes", summary="클래스 이름 목록")
def get_classes():
    return {"classes": CLASS_NAMES, "num_classes": NUM_CLASSES}


# ============================================================
# ## 엔드포인트 (3) — POST /infer  이미지 추론
#
# 요청: multipart/form-data  (file 필드에 이미지 첨부)
# 응답: {
#   "predicted_labels": ["A", "C"],       # 예측된 클래스 이름
#   "predicted_vector": [1,0,1,0,0,0,0],  # 이진 벡터
#   "scores": {"A":0.81, "B":0.12, ...},  # 클래스별 확률
#   "threshold": 0.5,
#   "filename": "uuid.jpg"                # 저장된 파일명
# }
# ============================================================
@app.post("/infer", summary="이미지 멀티레이블 추론")
async def infer(
    file: UploadFile = File(..., description="분류할 이미지 (jpg/png/webp)"),
):
    # 모델 로드 여부 확인
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. multilabel_vgg16.pth 파일을 먼저 생성하세요.",
        )

    img_bytes = await file.read()

    # 확장자 검증 + 저장
    saved_name = validate_and_save(img_bytes, file.filename or "upload.jpg")

    # 전처리
    tensor = bytes_to_tensor(img_bytes)

    # 추론
    result = run_inference(model, tensor)
    result["filename"] = saved_name

    return JSONResponse(content=result)


# ============================================================
# ## 직접 실행 진입점
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "multi_lable_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
