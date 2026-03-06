import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from pathlib import Path
from PIL import Image
import io
from torchvision import models, transforms

# 페이지 설정
st.set_page_config(
    page_title="지능형 민원처리 시스템",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 추가
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3em 0;
    }
    
    .main-title {
        text-align: center;
        color: #6c5ce7;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 0.3em;
    }
    
    .main-icon {
        text-align: center;
        font-size: 3.5em;
        margin-bottom: 0.5em;
    }
    
    .subtitle {
        text-align: center;
        color: #333;
        font-size: 0.95em;
        margin-bottom: 3em;
        line-height: 1.5;
    }
    
    .section-card {
        background: white;
        border-radius: 1em;
        padding: 2em;
        margin-bottom: 2em;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.1);
        border-top: 4px solid #6c5ce7;
    }
    
    .section-title {
        color: #6c5ce7;
        font-size: 1.4em;
        font-weight: 700;
        margin-bottom: 1.2em;
        margin-bottom: 1.2em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    
    .input-section {
        background: white;
        border-radius: 1em;
        padding: 2em;
        margin-bottom: 2em;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.1);
    }
    
    .input-label {
        color: #6c5ce7;
        font-size: 0.95em;
        font-weight: 600;
        margin-bottom: 0.5em;
        display: block;
    }
    
    .agency-container {
        display: flex;
        gap: 1em;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .agency-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 1em;
        padding: 1.5em;
        text-align: center;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.2);
        transition: transform 0.3s ease;
        max-width: 250px;
        flex: 1;
        min-width: 100px;
    }
    
    .agency-card:hover {
        transform: translateY(-5px);
    }
    
    .agency-icon {
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    
    .agency-name {
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 0.3em;
    }
    
    .agency-prob {
        font-size: 0.85em;
        opacity: 0.9;
    }
    
    .doc-item {
        background: #f8f9fa;
        padding: 1em;
        border-radius: 0.6em;
        margin-bottom: 0.8em;
        border-left: 3px solid #6c5ce7;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .doc-name {
        color: #333;
        font-weight: 500;
    }
    
    .doc-prob {
        background: #6c5ce7;
        color: white;
        padding: 0.3em 0.8em;
        border-radius: 2em;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .upload-area {
        border: 2px dashed #6c5ce7;
        border-radius: 1em;
        padding: 2em;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: #f0f0f7;
        border-color: #764ba2;
    }
    
    .file-item {
        background: white;
        padding: 1em;
        border-radius: 0.6em;
        margin-bottom: 0.8em;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #e0e0e0;
    }
    
    .status-complete {
        color: #27ae60;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    .status-incomplete {
        color: #e74c3c;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.6em;
        padding: 0.7em 2em;
        font-weight: 600;
        font-size: 1em;
        transition: transform 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    .empty-state {
        text-align: center;
        color: #999;
        padding: 2em;
        font-size: 0.95em;
    }
    
    .document-card {
        background: white;
        border-radius: 1em;
        padding: 1.5em;
        margin-bottom: 1.2em;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .document-card:hover {
        border-color: #6c5ce7;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.15);
    }
    
    .document-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2em;
        padding-bottom: 1em;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .document-name-section {
        display: flex;
        align-items: center;
        gap: 0.8em;
    }
    
    .document-title {
        font-weight: 700;
        font-size: 1.1em;
        color: #333;
    }
    
    .document-prob-badge {
        background: #667eea;
        color: white;
        padding: 0.4em 0.8em;
        border-radius: 1em;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .document-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.2em;
    }
    
    .upload-section {
        display: flex;
        flex-direction: column;
    }
    
    .upload-label {
        color: #6c5ce7;
        font-size: 0.9em;
        font-weight: 600;
        margin-bottom: 0.8em;
    }
    
    .document-upload-box {
        border: 2px dashed #6c5ce7;
        border-radius: 0.8em;
        padding: 1.5em;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .document-upload-box:hover {
        background: #f0f0f7;
        border-color: #764ba2;
    }
    
    .upload-icon {
        font-size: 2em;
        margin-bottom: 0.5em;
    }
    
    .upload-text {
        font-size: 0.85em;
        color: #666;
    }
    
    .status-section {
        display: flex;
        flex-direction: column;
    }
    
    .status-label {
        color: #6c5ce7;
        font-size: 0.9em;
        font-weight: 600;
        margin-bottom: 0.8em;
    }
    
    .status-empty {
        border: 1px solid #e0e0e0;
        border-radius: 0.8em;
        padding: 1.5em;
        text-align: center;
        background: #fafafa;
        color: #999;
        font-size: 0.9em;
    }
    
    .status-item {
        background: #f8f9fa;
        padding: 1em;
        border-radius: 0.6em;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 4px solid #6c5ce7;
    }
    
    .file-name {
        color: #333;
        font-weight: 500;
        word-break: break-word;
    }
    
    .status-badge {
        display: flex;
        align-items: center;
        gap: 0.4em;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .status-badge.success {
        color: #27ae60;
    }
    
    .status-badge.fail {
        color: #e74c3c;
    }
    
    @media (max-width: 768px) {
        .document-content {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "best_model"
# 초기화(없으면 만들어두기)
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# 모델 및 토크나이저 캐시
@st.cache_resource
def load_model_and_tokenizer():
    """모델과 토크나이저 로드"""    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    
    # 라벨 로드
    with open(DATA_DIR / "multilabel_classes.json", "r", encoding="utf-8") as f:
        label_cols = json.load(f)
    
    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        "klue/roberta-base",
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )
    
    # 저장된 가중치 로드
    model_path = MODEL_DIR / "multiLabel_best_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, label_cols, device

# 민원 분류 함수
def classify_complaint(text, model, tokenizer, label_cols, device, threshold=0.4):
    """민원 텍스트 분류"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    # 예측 결과 추출
    predictions = []
    for col, prob in zip(label_cols, probs):
        if prob >= threshold:
            predictions.append((col, float(prob)))
    
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return predictions

# 라벨에서 기관과 문서 분류
def categorize_labels(predictions):
    """예측된 라벨을 기관과 필요 서류로 분류"""
    agencies = {}
    depart = {}
    compla = {}
    documents = {}
    
    for label, prob in predictions:
        if label.startswith("기관:"):
            agency = label.replace("기관:", "")
            agencies[agency] = prob
        elif label.startswith("부서:"):
            dept = label.replace("부서:", "")
            depart[dept] = prob
        elif label.startswith("민원명:"):
            comp = label.replace("민원명:", "")
            compla[comp] = prob
        elif label.startswith("문서:"):
            doc = label.replace("문서:", "")
            documents[doc] = prob
    
    return agencies, depart, compla, documents

# 부서 아이콘 매핑
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

# 필요 서류 기본 템플릿
REQUIRED_DOCS_TEMPLATE = {
    "여권신청서": False,
    "운전면허증": False,
    "전입신고서": False,
    "확정일자": False,
    "주민등록증": False
}

# 필수 문서 요소 모델 불러오기
def load_required_docs_model():
    """필수 문서 요소 모델 로드"""    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280,5) 

    model.load_state_dict(torch.load(MODEL_DIR / "document_best_model.pt", map_location=device, weights_only=False))      

    transform_test = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )

    model.to(device)
    model.eval()

    return model, transform_test, device



# 메인 UI인터페이스
def main():
    st.markdown('<div class="main-title">지능형 민원처리 시스템</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">신뢰감 주는 타이틀과 함께 서비스의 정체성을<br/>명확히 전달하는 상담 영역입니다.</div>',
        unsafe_allow_html=True
    )
    
    # 모델 로드
    try:
        model, tokenizer, label_cols, device = load_model_and_tokenizer()
        docu_model, transform_test, docu_device = load_required_docs_model()
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        return
    
    # 민원 입력 섹션
    st.markdown('<label class="input-label">📝 민원 내용을 자유롭게 입력해주세요</label>', unsafe_allow_html=True)
    complaint_text = st.text_area(
        label="complaint",
        placeholder="예: 여권을 만들고 싶어요",
        height=120,
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("✨ 작성 완료", use_container_width=True)
    
    # 분석 결과
    if analyze_btn:
        if complaint_text.strip():
            with st.spinner("🔄 민원 분석 중..."):
                st.session_state.predictions = classify_complaint(
                    complaint_text, 
                    model, 
                    tokenizer, 
                    label_cols, 
                    device
                )
        else:
            st.warning("민원 내용을 입력해주세요.")

    predictions = st.session_state.predictions    
    # 추천 부서 섹션
    if predictions:
        agencies, depart, compla, documents = categorize_labels(predictions)
        
        if agencies or depart or compla:
            st.markdown('<div class="section-title">🏢 추천 부서</div>', unsafe_allow_html=True)
            
            # HTML flexbox 컨테이너 시작
            agency_html = '<div class="agency-container">'
            
            # 첫 번째 카드: 기관
            if agencies:
                agency, a_prob = list(agencies.items())[0]
                a_icon = "🏛️"
                for key, value in AGENCY_ICONS.items():
                    if key in agency:
                        a_icon = value
                        break
                
                agency_html += f"""
                    <div class="agency-card">
                        <div class="agency-icon">{a_icon}</div>
                        <div class="agency-name">{agency}</div>
                        <div class="agency-prob">{a_prob:.1%}</div>
                    </div>"""
            
            # 두 번째 카드: 부서
            if depart:
                depart, d_prob = list(depart.items())[0]
                d_icon = "📂"
                for key, value in AGENCY_ICONS.items():
                    if key in depart:   
                        d_icon = value
                        break
                agency_html += f"""
                    <div class="agency-card" style="background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);">
                        <div class="agency-icon">{d_icon}</div>
                        <div class="agency-name">{depart}</div>
                        <div class="agency-prob">{d_prob:.1%}</div>
                    </div>"""
            
            # 세 번째 카드: 민원
            if compla:
                compla, c_prob = list(compla.items())[0]
                c_icon = "📝"
                for key, value in AGENCY_ICONS.items():
                    if key in compla:
                        c_icon = value
                        break
                agency_html += f"""
                    <div class="agency-card" style="background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);">
                        <div class="agency-icon">{c_icon}</div>
                        <div class="agency-name">{compla}</div>
                        <div class="agency-prob">{c_prob:.1%}</div>
                    </div>"""
            
            # 컨테이너 닫기
            agency_html += '</div>'
            
            st.markdown(agency_html, unsafe_allow_html=True)

        else:
            st.markdown('<div class="empty-state">🔍 분석 결과가 없습니다. 민원 내용을 좀 더 구체적으로 작성해주세요.</div>', unsafe_allow_html=True)
        
        # 필요 서류 섹션
        if documents:
            st.markdown('<div class="section-title">📋 필요 서류 목록</div>', unsafe_allow_html=True)
            
            for doc, prob in list(documents.items())[:6]:
                st.markdown(f"""
                <div class="doc-item">
                    <span class="doc-name">☑️ {doc}</span>
                    <span class="doc-prob">{prob:.0%}</span>
                </div>
                """, unsafe_allow_html=True)            
    
    # 문서 업로드 및 검증 섹션
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 서류 업로드 및 검증</div>', unsafe_allow_html=True)
    
    predictions = st.session_state.predictions
    
    # 필요 서류가 있으면 문서별로 표시
    if predictions and documents:
        # 세션 상태에 파일 저장 영역 초기화
        if "uploaded_docs" not in st.session_state:
            st.session_state.uploaded_docs = {}
        
        # 문서별 카드 생성
        for idx, (doc, prob) in enumerate(list(documents.items())[:6]):
            doc_key = f"doc_{idx}_{doc}"
            
            # HTML 카드 시작
            st.markdown(f"""
            <div class="document-card">
                <div class="document-header">
                    <div class="document-name-section">
                        <span style="font-size: 1.3em;">📄</span>
                        <div>
                            <div class="document-title">{doc}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 업로드와 상태를 두 칼럼으로 배치
            col_up, col_status = st.columns([1, 1])
            
            with col_up:
                uploaded_file = st.file_uploader(
                    f"파일 선택 ({doc})",
                    type=["jpg", "jpeg", "png", "pdf"],
                    key=doc_key,
                    label_visibility="collapsed"
                )
                
                # 파일 업로드 시 세션 상태에 저장
                if uploaded_file:
                    st.session_state.uploaded_docs[doc_key] = uploaded_file
            
            with col_status:
                # 검증 상태 표시
                if doc_key in st.session_state.uploaded_docs:
                    uploaded_file = st.session_state.uploaded_docs[doc_key]
                    file_name = uploaded_file.name
                    is_valid = uploaded_file.size > 0
                    status_icon = "✅" if is_valid else "❌"
                    status_text = "정상" if is_valid else "미흡"
                    status_class = "success" if is_valid else "fail"
                    
                    st.markdown(f"""
                    <div style="padding: 1em; border-radius: 0.6em; background: #f8f9fa;">
                        <div style="color: #666; font-size: 0.85em; margin-bottom: 0.5em; word-break: break-word;">📁 {file_name}</div>
                        <div class="status-badge {status_class}">{status_icon} {status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-empty">📁 파일을 선택하면<br/>검증 결과가 표시됩니다</div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 1.5em 0; border: none; border-top: 1px solid #f0f0f0;'>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">📋 민원 내용을 먼저 분석해주세요</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
