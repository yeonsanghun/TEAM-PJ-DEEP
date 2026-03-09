import base64
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

# =====================================================================
# Config
# =====================================================================
NUM_CLASSES = 14
THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "best_model"
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

DOC_CLASSES = [
    '여권', '여권신청서', '운전면허증', '임대차계약서', '전입신고서', '주민등록등본', '주민등록증', '확정일자신청서'
]

# -----------------------------------------------------
# Pydantic models for text/document classification API
# -----------------------------------------------------

class ComplaintRequest(BaseModel):
    """민원 분류 요청 스키마"""
    text: str


class ClassifyResponse(BaseModel):
    """민원 분류 응답 스키마"""
    agencies: Dict[str, float]
    departments: Dict[str, float]
    complaints: Dict[str, float]
    documents: Dict[str, float]
    predictions: List[Tuple[str, float]]


class DocumentClassifyResponse(BaseModel):
    """문서 이미지 분류 응답 스키마"""
    document_class: str
    confidence: float
    filename: str

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 이미 정의된 경로 변수는 위에서 생성됨

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 전역 모델 (서버 시작 시 1회 로드)
# =====================================================================
# ------ EfficientNetB4 for document-field detection ------
_model: nn.Module | None = None

def get_model() -> nn.Module:
    global _model
    if _model is None:
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(1792, NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_DIR / "best_model_efficiB4.pth", map_location="cpu"))
        m.eval()
        _model = m
    return _model

# ------ helper models for text/document classification ------
# caching globals for models/tokenizer/labels
_complaint_model: nn.Module | None = None
_complaint_tokenizer = None
_label_cols: List[str] | None = None
_doc_model: nn.Module | None = None


def get_complaint_tokenizer():
    """로버타 토크나이저를 전역 캐시로 반환합니다."""
    global _complaint_tokenizer
    if _complaint_tokenizer is None:
        _complaint_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    return _complaint_tokenizer


def get_label_cols() -> List[str]:
    """라벨 목록을 파일에서 읽어와 캐시하여 반환합니다."""
    global _label_cols
    if _label_cols is None:
        with open(DATA_DIR / "multilabel_classes.json", "r", encoding="utf-8") as f:
            _label_cols = json.load(f)
    return _label_cols


def get_complaint_model() -> nn.Module:
    """ROBERTA 기반 멀티라벨 분류 모델을 로드하고 캐시하여 반환합니다."""
    global _complaint_model
    if _complaint_model is None:
        label_cols = get_label_cols()
        m = AutoModelForSequenceClassification.from_pretrained(
            "klue/roberta-base",
            num_labels=len(label_cols),
            problem_type="multi_label_classification"
        )
        model_path = MODEL_DIR / "multiLabel_best_model.pt"
        if model_path.exists():
            m.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        m.to(device)
        m.eval()
        _complaint_model = m
    return _complaint_model


def get_doc_model() -> nn.Module:
    """EfficientNet-B0 문서 분류 모델을 로드하고 캐시하여 반환합니다."""
    global _doc_model
    if _doc_model is None:
        m = models.efficientnet_b0(pretrained=True)
        m.classifier[1] = nn.Linear(1280, len(DOC_CLASSES))
        model_path = MODEL_DIR / "document_best_model.pt"
        if model_path.exists():
            m.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        m.to(device)
        m.eval()
        _doc_model = m
    return _doc_model

# these will be stored on app.state in lifespan



# =====================================================================
# 추론 + GradCAM++ for efficientnet B4 fields
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
# 로버타 텍스트/문서 분류 helper 함수
# =====================================================================

def classify_complaint(
    text: str,
    model,
    tokenizer,
    label_cols: List[str],
    threshold: float = 0.4
) -> List[Tuple[str, float]]:
    # ... (content unchanged from fastapi_main2)
    # 텍스트 토큰화
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # 모델 추론
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    # 임계값 이상의 예측 추출
    predictions = []
    for col, prob in zip(label_cols, probs):
        if prob >= threshold:
            predictions.append((col, float(prob)))
    
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return predictions


def categorize_labels(predictions: List[Tuple[str, float]]) -> Tuple[Dict, Dict, Dict, Dict]:
    agencies = {}
    departments = {}
    complaints = {}
    documents = {}
    
    for label, prob in predictions:
        if label.startswith("기관:"):
            agency = label.replace("기관:", "")
            agencies[agency] = prob
        elif label.startswith("부서:"):
            dept = label.replace("부서:", "")
            departments[dept] = prob
        elif label.startswith("민원명:"):
            comp = label.replace("민원명:", "")
            complaints[comp] = prob
        elif label.startswith("문서:"):
            doc = label.replace("문서:", "")
            documents[doc] = prob
    
    return agencies, departments, complaints, documents


# =====================================================================
# FastAPI 앱 및 생명주기 관리
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 필요한 리소스를 준비합니다.

    - EfficientNetB4 모델(기존)과
    - ROBERTA 텍스트 분류 모델 및
    - EfficientNetB0 문서 분류 모델
      을 서버 기동 시 한 번 로드하여 app.state에 저장합니다.
    """
    print("========== 서버 시작: 모델 로드 ==============")

    # 기존 B4 모델을 캐시해 둡니다 (get_model 내부에서 처리).
    get_model()

    # -------- ROBERTA 텍스트 분류 모델 --------
    print("ROBERTA tokenizer/model/labels 로드 중...")
    tokenizer = get_complaint_tokenizer()
    app.state.tokenizer = tokenizer

    label_cols = get_label_cols()
    app.state.label_cols = label_cols

    app.state.complaint_model = get_complaint_model()

    # -------- EfficientNetB0 문서 분류 모델 --------
    print("문서 분류 모델(EfficientNetB0) 로드 중...")
    app.state.doc_model = get_doc_model()

    # 문서 분류용 전처리 파이프라인
    transform_doc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    app.state.transform_doc = transform_doc

    print(f"사용 디바이스: {device}")
    print("========== 모델 로드 완료 =============="
          )

    yield

    print("서버 종료 중...")


app = FastAPI(
    title="지능형 민원처리 시스템 API",
    description="텍스트 기반 민원 분류, 문서/이미지 필드 예측",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """서버 상태 및 제공되는 서비스 목록 반환"""
    return {
        "status": "OK",
        "service": "지능형 민원처리 시스템 API",
        "models": {
            "field_detection": "EfficientNet-B4",
            "text_classification": "ROBERTA (klue/roberta-base)",
            "document_classification": "EfficientNet-B0"
        },
        "labels": LABEL_COLS,
        "endpoints": {
            "GET /": "서버 상태 확인",
            "POST /efficiNetB4": "이미지 업로드 → 필드 예측 + GradCAM++",
            "POST /classify": "민원 텍스트 분류",
            "POST /classify-document": "문서 이미지 분류",
            "GET /icons": "부서별 아이콘 목록",
            "GET /document-classes": "분류 가능한 문서 클래스",
            "GET /info": "시스템 정보"
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

@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(
    request: Request,
    complaint: ComplaintRequest
):
    """
    민원 텍스트를 분류하고 필요한 기관/부서/문서를 반환합니다.
    
    요청 예시:
    ```json
    {
        "text": "여권을 만들고 싶어요"
    }
    ```
    
    응답 예시:
    ```json
    {
        "agencies": {"기관명": 0.95, ...},
        "departments": {"부서명": 0.92, ...},
        "complaints": {"민원명": 0.88, ...},
        "documents": {"여권신청서": 0.91, ...},
        "predictions": [["기관:외교부", 0.95], ...]
    }
    ```
    """
    try:
        # 모델 가져오기
        model = request.app.state.complaint_model
        tokenizer = request.app.state.tokenizer
        label_cols = request.app.state.label_cols
        
        if not complaint.text.strip():
            return {
                "error": "텍스트 입력이 없습니다",
                "agencies": {},
                "departments": {},
                "complaints": {},
                "documents": {},
                "predictions": []
            }
        
        # 텍스트 분류
        predictions = classify_complaint(
            complaint.text,
            model,
            tokenizer,
            label_cols
        )
        
        # 라벨 분류
        agencies, departments, complaints_dict, documents = categorize_labels(predictions)
        
        return ClassifyResponse(
            agencies=agencies,
            departments=departments,
            complaints=complaints_dict,
            documents=documents,
            predictions=predictions
        )
    
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return {
            "error": str(e),
            "agencies": {},
            "departments": {},
            "complaints": {},
            "documents": {},
            "predictions": []
        }
    
@app.post("/classify-document", response_model=DocumentClassifyResponse)
async def classify_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    문서 이미지를 분류합니다.
    
    지원 포맷: jpg, jpeg, png, webp
    분류 클래스: 여권신청서, 운전면허증, 전입신고서, 주민등록증, 확정일자
    
    요청: multipart/form-data (file 필드에 이미지 파일)
    
    응답 예시:
    ```json
    {
        "document_class": "여권신청서",
        "confidence": 0.95,
        "filename": "document.jpg"
    }
    ```
    """
    try:
        # 파일 읽기
        img_bytes = await file.read()
        
        if not img_bytes:
            return {
                "error": "파일이 비어있습니다",
                "document_class": None,
                "confidence": 0.0,
                "filename": file.filename
            }
        
        # PIL Image로 변환
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 모델 가져오기
        model = request.app.state.doc_model
        transform = request.app.state.transform_doc
        
        # 이미지 전처리 및 추론
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            class_idx = torch.argmax(probs, dim=1).item()
            confidence = float(probs[0, class_idx])
        
        return DocumentClassifyResponse(
            document_class=DOC_CLASSES[class_idx],
            confidence=confidence,
            filename=file.filename
        )
    
    except Exception as e:
        print(f"문서 분류 에러: {str(e)}")
        return {
            "error": str(e),
            "document_class": None,
            "confidence": 0.0,
            "filename": file.filename
        }