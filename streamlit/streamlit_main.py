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
    
    .agency-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 1em;
        padding: 1.5em;
        text-align: center;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.2);
        transition: transform 0.3s ease;
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
    </style>
""", unsafe_allow_html=True)

# 데이터 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "연상훈"

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
def classify_complaint(text, model, tokenizer, label_cols, device, threshold=0.3):
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
    documents = {}
    
    for label, prob in predictions:
        if label.startswith("기관:"):
            agency = label.replace("기관:", "")
            agencies[agency] = prob
        elif label.startswith("부서:"):
            dept = label.replace("부서:", "")
            depart[dept] = prob
        elif label.startswith("문서:"):
            doc = label.replace("문서:", "")
            documents[doc] = prob
    
    return agencies, depart, documents

# 부서 아이콘 매핑
AGENCY_ICONS = {
    "교통": "🚗",
    "복지": "❤️",
    "환경": "🌍",
    "여권": "🛂️",
    "세무": "💼",
    "출입국": "✈️",
    "기본주소": "🏠"
}

# 필요 서류 기본 템플릿
REQUIRED_DOCS_TEMPLATE = {
    "신분증": False,
    "신청서": False,
    "인감도장": False,
    "매매계약서": False,
    "자동차등록증": False,
    "여권신청서": False,
}

# 메인 UI인터페이스
def main():
    st.markdown('<div class="main-icon">🏛️</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">지능형 민원처리 시스템</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">신뢰감 주는 타이틀과 함께 서비스의 정체성을<br/>명확히 전달하는 상담 영역입니다.</div>',
        unsafe_allow_html=True
    )
    
    # 모델 로드
    try:
        model, tokenizer, label_cols, device = load_model_and_tokenizer()
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
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("✨ 작성 완료", use_container_width=True)
    
    # 분석 결과
    predictions = None
    if analyze_btn:
        if complaint_text.strip():
            with st.spinner("🔄 민원 분석 중..."):
                predictions = classify_complaint(
                    complaint_text, 
                    model, 
                    tokenizer, 
                    label_cols, 
                    device
                )
        else:
            st.warning("민원 내용을 입력해주세요.")
    
    # 추천 부서 섹션
    if predictions:
        agencies, depart, documents = categorize_labels(predictions)
        
        if agencies:
            st.markdown('<div class="section-title">🏢 추천 부서</div>', unsafe_allow_html=True)
            
            agency_list = list(agencies.items())[:3]
            agency_cols = st.columns(len(agency_list))
            
            for col, (agency, prob) in zip(agency_cols, agency_list):
                with col:
                    # 아이콘 찾기
                    icon = "🏛️"
                    for key, value in AGENCY_ICONS.items():
                        if key in agency:
                            icon = value
                            break
                    
                    st.markdown(f"""
                    <div class="agency-card">
                        <div class="agency-icon">{icon}</div>
                        <div class="agency-name">{agency}</div>
                        <div class="agency-prob">{prob:.1%}</div>
                    """, unsafe_allow_html=True)

                    depart_list = list(depart.items())[:3]
                    depart_cols = st.columns(len(depart_list))
                    for d_col, (dept, d_prob) in zip(depart_cols, depart_list):
                        with d_col:
                            st.markdown(f"""
                            <div class="agency-card" style="background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);">
                                <div class="agency-icon">📂</div>
                                <div class="agency-name">{dept}</div>
                                <div class="agency-prob">{d_prob:.1%}
                            </div>""", unsafe_allow_html=True)

                    st.markdown(f"</div>", unsafe_allow_html=True)
            
        
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
    st.markdown('<div class="section-title">📤 서류 업로드 및 파일 관리</div>', unsafe_allow_html=True)
    
    col_upload, col_result = st.columns([1, 1])
    
    with col_upload:
        st.markdown('<label class="input-label">서류 선택 또는 촬영</label>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "파일을 선택하세요",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    with col_result:
        st.markdown('<label class="input-label">자동 검증 및 최종 상태 표시</label>', unsafe_allow_html=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                is_valid = uploaded_file.size > 0
                status_icon = "✅" if is_valid else "❌"
                status_text = "정상" if is_valid else "미흡"
                status_class = "status-complete" if is_valid else "status-incomplete"
                
                st.markdown(f"""
                <div class="file-item">
                    <span>{file_name}</span>
                    <span class="{status_class}">{status_icon} {status_text}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">📁 파일을 업로드하면 검증 결과가 표시됩니다</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
