import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
import streamlit as st

# =====================================================================
# 모델별 설정 레지스트리
# =====================================================================
LABEL_COLS = [
    "전입자_성명", "전입자_주민등록번호", "전입자_연락처", "전입자_서명도장",
    "전_시도", "전_시군구", "현_세대주성명", "현_연락처", "현_서명도장",
    "현_주소", "전입사유_체크", "우편물서비스_동의체크", "신청인_성명", "신청인_서명도장",
]

MODEL_REGISTRY = {
    "ConvNeXt-S (전입신고서 Multi-Label) 1회차": {
        "model_path": r"..\document_forms_source\checkpoints\best_model-data2-1000_20260309_Best_Val_Loss_0.1439.pth",
        "type": "multilabel",
        "num_classes": 14,
        "labels": LABEL_COLS,
        "threshold": 0.5,
        # 1회차: ImageNet 기준 정규화
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "ConvNeXt-S (전입신고서 Multi-Label) 2회차": {
        "model_path": r"..\document_forms_source\checkpoints\best_model-data2-4000-unfreezed_20260309-val_loss_0.0087.pth",
        "type": "multilabel",
        "num_classes": 14,
        "labels": LABEL_COLS,
        "threshold": 0.5,
        # 2회차: 커스텀 정규화
        "mean": [0.9367, 0.9364, 0.9358],
        "std": [0.0957, 0.0964, 0.0963],
    },
    "ConvNeXt-S (전입신고서 Multi-Label) 3회차(작성해야함)": {
        "model_path": r"..\document_forms_source\checkpoints\best_model-data2-4000-unfreezed_20260309-val_loss_0.0087.pth",
        "type": "multilabel",
        "num_classes": 14,
        "labels": LABEL_COLS,
        "threshold": 0.5,
        # 3회차: 커스텀 정규화
        "mean": [0.9367, 0.9364, 0.9358],
        "std": [0.0957, 0.0964, 0.0963],
    },
}

# =====================================================================
# 전처리 함수 (모델 설정에 따라 동적 생성)
# =====================================================================
def get_transform(config: dict):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])

# =====================================================================
# 모델 로드
# =====================================================================
@st.cache_resource
def load_convnext_small(model_path: str, num_classes: int):
    model = models.convnext_small(weights=None)
    model.classifier[2] = nn.Linear(768, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# =====================================================================
# GradCAM++ (전처리를 인자로 받도록 수정)
# =====================================================================
def compute_gradcam_pp(model, image: Image.Image, transform) -> np.ndarray:
    for param in model.parameters():
        param.requires_grad = True

    input_tensor = transform(image).unsqueeze(0)
    pred = model(input_tensor)
    pred_class = int(torch.sigmoid(pred).argmax().item())

    cam_pp = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])
    gradcam_map = cam_pp(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_class)],
    )[0]

    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    return show_cam_on_image(rgb_img, gradcam_map, use_rgb=True)

# =====================================================================
# 예측
# =====================================================================
def predict_multilabel(model, tensor: torch.Tensor, config: dict):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)[0]
    preds = (probs >= config["threshold"]).tolist()
    labels = config["labels"]
    return [
        {"label": labels[i], "prob": float(probs[i]), "pred": preds[i]}
        for i in range(len(labels))
    ]

# =====================================================================
# Streamlit UI
# =====================================================================
st.set_page_config(page_title="모델 테스터", layout="wide")
st.title("📋 학습 모델 테스터")

model_name = st.selectbox("테스트할 모델 선택", list(MODEL_REGISTRY.keys()))
config = MODEL_REGISTRY[model_name]

# 모델 파일 경로 설정
model_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), config["model_path"]))
if not os.path.exists(model_abs):
    st.error(f"모델 파일을 찾을 수 없습니다: `{model_abs}`")
    st.stop()

# 현재 모델의 전처리(transform) 생성
current_transform = get_transform(config)

uploaded_file = st.file_uploader("테스트 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col2:
        with st.spinner("모델 로딩 중..."):
            model = load_convnext_small(model_abs, config["num_classes"])

        # 전처리 적용
        input_tensor = current_transform(image).unsqueeze(0)

        st.subheader("예측 결과")
        results = predict_multilabel(model, input_tensor, config)

        filled = [r for r in results if r["pred"]]
        empty = [r for r in results if not r["pred"]]

        st.markdown(f"**현재 적용된 정규화값**: \nMean: {config['mean']}, Std: {config['std']}")
        
        st.markdown("**✅ 기입된 필드**")
        if filled:
            for r in filled:
                st.progress(r["prob"], text=f"{r['label']} ({r['prob'] * 100:.1f}%)")
        else:
            st.info("기입된 필드 없음")

        st.markdown("**❌ 미기입 필드**")
        if empty:
            for r in empty:
                st.progress(r["prob"], text=f"{r['label']} ({r['prob'] * 100:.1f}%)")

    with col1:
        st.image(image, caption="업로드한 이미지")
        with st.spinner("GradCAM++ 생성 중..."):
            # 동적 생성된 transform 전달
            gradcam_img = compute_gradcam_pp(model, image, current_transform)
        st.image(gradcam_img, caption="GradCAM++ 히트맵")