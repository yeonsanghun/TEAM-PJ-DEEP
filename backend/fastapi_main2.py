# ============================================================
# FastAPI 기반 지능형 민원처리 시스템
# streamlit_main.py의 핵심 로직을 FastAPI로 구현
# ============================================================
#
# ┌─────────────────────────────────────────────────────────┐
# │              전체 실행 흐름 (Execution Flow)              │
# │                                                         │
# │  [서버 기동 시]                                           │
# │   Step 1. 라이브러리 임포트 및 전역 디바이스 설정           │
# │   Step 2. lifespan — 모델 로드 (Startup)                 │
# │                                                         │
# │  [민원 분류 요청 시 — POST /classify 호출 순서]           │
# │   Step 3. 민원 텍스트 입력 수신                          │
# │   Step 4. ROBERTA 모델로 분류 추론                        │
# │   Step 5. 예측 라벨을 기관/부서/민원/문서로 분류            │
# │   Step 6. 최종 JSON 응답 생성 반환                        │
# │                                                         │
# │  [문서 이미지 분류 요청 시 — POST /classify-document]     │
# │   Step 3. 이미지 파일 수신                               │
# │   Step 4. EfficientNetB0 모델로 분류 추론                 │
# │   Step 5. 최종 JSON 응답 생성 반환                        │
# │                                                         │
# │  [서버 종료 시]                                           │
# │   Step 2. lifespan — 자원 해제 (Shutdown)               │
# └─────────────────────────────────────────────────────────┘

import io
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from pydantic import BaseModel

from fastapi import FastAPI, Request, UploadFile, File


# ============================================================
# ## Step 1. 설정 및 경로 정의
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "best_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ## Pydantic 모델 정의 (요청/응답 스키마)
# ============================================================

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


# ============================================================
# ## 부서 아이콘 매핑 및 상수 정의
# ============================================================

AGENCY_ICONS = {
    "민원": "📄",
    "민원접수": "📝",
    "민원처리": "✅",
    "처리중": "⏳",
    "보완요청": "⚠️",
    "서류확인": "🔍",
    "서류제출": "📬",
    "문서분류": "🗂️",
    "상담": "☎️",
    "안내": "📢",
    "교통": "🚗",
    "복지": "❤️",
    "환경": "🌍",
    "여권": "🛂",
    "세무": "💼",
    "출입국": "✈️",
    "주소": "🏠",
    "주민등록": "🪪",
    "건축": "🏗️",
    "부동산": "🏢",
    "가족관계": "👨‍👩‍👧",
    "출생": "👶",
    "사망": "🕯️",
    "혼인": "💍",
    "이혼": "📑",
    "병무": "🎖️",
    "교육": "🎓",
    "보건": "🏥",
    "의료": "💊",
    "고용": "💼",
    "노동": "🛠️",
    "사업자": "🏪",
    "법인": "🏛️",
    "인허가": "📜",
    "증명서": "📃",
    "신고": "📢",
    "신청": "📝",
    "발급": "🖨️",
    "재발급": "🔁",
    "납부": "💳",
    "조회": "🔎",
    "예약": "📅",
    "문의": "❓",
    "불편": "😟",
    "신고접수": "🚨",
    "안전": "🦺",
    "소방": "🔥",
    "경찰": "👮",
    "도로": "🛣️",
    "주차": "🅿️",
    "쓰레기": "🗑️",
    "상하수도": "🚰",
    "전기": "⚡",
    "동물": "🐾",
    "식품": "🍽️"
}

DOC_CLASSES = [
    "여권신청서",
    "운전면허증",
    "전입신고서",
    "주민등록증",
    "확정일자"
]


# ============================================================
# ## Step 2. 서버 생명주기 관리 — 모델 로드(Startup)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 생명주기를 관리합니다.
    - yield 이전: 서버 시작 시 모델 로드
    - yield 이후: 서버 종료 시 리소스 정리
    """
    print("========== 모델 불러오기 시작 ==============")
    
    # -------- ROBERTA 모델 로드 --------
    print("ROBERTA 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    app.state.tokenizer = tokenizer
    
    print("라벨 정보 로드 중...")
    with open(DATA_DIR / "multilabel_classes.json", "r", encoding="utf-8") as f:
        label_cols = json.load(f)
    app.state.label_cols = label_cols
    
    print("ROBERTA 모델 로드 중...")
    complaint_model = AutoModelForSequenceClassification.from_pretrained(
        "klue/roberta-base",
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )
    
    model_path = MODEL_DIR / "multiLabel_best_model.pt"
    if model_path.exists():
        complaint_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
    
    complaint_model.to(device)
    complaint_model.eval()
    app.state.complaint_model = complaint_model
    
    # -------- EfficientNetB0 모델 로드 --------
    print("문서 분류 모델(EfficientNetB0) 로드 중...")
    doc_model = models.efficientnet_b0(pretrained=True)
    doc_model.classifier[1] = nn.Linear(1280, 5)
    
    doc_model_path = MODEL_DIR / "document_best_model.pt"
    if doc_model_path.exists():
        doc_model.load_state_dict(
            torch.load(doc_model_path, map_location=device, weights_only=False)
        )
    
    doc_model.to(device)
    doc_model.eval()
    app.state.doc_model = doc_model
    
    # 문서 분류용 전처리 파이프라인
    transform_doc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    app.state.transform_doc = transform_doc
    
    print(f"사용 디바이스: {device}")
    print("========== 모델 불러오기 완료 ==============\n")
    
    yield  # 서버가 요청을 처리하는 구간
    
    print("서버 종료 중...")


# ============================================================
# ## FastAPI 앱 인스턴스 생성
# ============================================================

app = FastAPI(
    title="지능형 민원처리 시스템 API",
    description="텍스트 기반 민원 분류 및 필요 서류 예측",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# ## 핵심 로직 함수들
# ============================================================

def classify_complaint(
    text: str,
    model,
    tokenizer,
    label_cols: List[str],
    threshold: float = 0.4
) -> List[Tuple[str, float]]:
    """
    민원 텍스트를 ROBERTA 모델로 분류합니다.
    
    Args:
        text: 분류할 민원 텍스트
        model: ROBERTA 분류 모델
        tokenizer: 토크나이저
        label_cols: 라벨 목록
        threshold: 예측 확률 임계값
    
    Returns:
        [(라벨명, 확률), ...] 형태의 예측 결과
    """
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
    """
    예측된 라벨을 기관, 부서, 민원명, 필요 서류로 분류합니다.
    
    Args:
        predictions: [(라벨, 확률), ...] 형태의 예측 결과
    
    Returns:
        (기관_dict, 부서_dict, 민원명_dict, 서류_dict)
    """
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


# ============================================================
# ## API 엔드포인트
# ============================================================

@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "status": "OK",
        "message": "지능형 민원처리 시스템 API가 실행 중입니다.",
        "endpoints": {
            "GET /": "서버 상태 확인",
            "POST /classify": "민원 텍스트 분류",
            "POST /classify-document": "문서 이미지 분류",
            "GET /docs": "Swagger UI API 문서",
            "GET /redoc": "ReDoc API 문서"
        }
    }


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


# ============================================================
# ## 추가 유틸리티 엔드포인트
# ============================================================

@app.get("/icons")
def get_icons():
    """부서별 아이콘 매핑 정보 반환"""
    return AGENCY_ICONS


@app.get("/document-classes")
def get_document_classes():
    """분류 가능한 문서 클래스 목록 반환"""
    return {
        "classes": DOC_CLASSES,
        "count": len(DOC_CLASSES)
    }


@app.get("/info")
def get_info():
    """시스템 정보 반환"""
    return {
        "name": "지능형 민원처리 시스템",
        "version": "1.0.0",
        "device": str(device),
        "models": {
            "text_classification": "ROBERTA (klue/roberta-base)",
            "document_classification": "EfficientNetB0"
        },
        "data_paths": {
            "models": str(MODEL_DIR),
            "data": str(DATA_DIR)
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # 로컬에서 실행할 경우
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
