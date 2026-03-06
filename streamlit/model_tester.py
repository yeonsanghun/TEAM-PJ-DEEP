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
# 새 모델 추가 시 이 딕셔너리에만 항목을 추가하면 된다.
# =====================================================================
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

MODEL_REGISTRY = {
    "EfficientNet-B4 (전입신고서 Multi-Label)": {
        "model_path": "../efficiNetB4/best_model_efficiB4.pth",
        "type": "multilabel",
        "num_classes": 14,
        "labels": LABEL_COLS,
        "threshold": 0.5,
    },
}


# =====================================================================
# 모델 로드
# =====================================================================
@st.cache_resource
def load_efficientnet_b4(model_path: str, num_classes: int):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(1792, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def load_model(config: dict):
    return load_efficientnet_b4(model_abs, config["num_classes"])


# =====================================================================
# 이미지 전처리
# =====================================================================
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def preprocess(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0)


# =====================================================================
# GradCAM++
# =====================================================================
def compute_gradcam_pp(model, image: Image.Image) -> np.ndarray:
    """GradCAM++ 오버레이 이미지(H×W×3 uint8 ndarray)를 반환한다."""
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
st.caption("best 모델을 로드하여 이미지를 테스트합니다.")

# 모델 선택
model_name = st.selectbox("테스트할 모델 선택", list(MODEL_REGISTRY.keys()))
config = MODEL_REGISTRY[model_name]

# 모델 파일 존재 확인
model_abs = os.path.abspath(
    os.path.join(os.path.dirname(__file__), config["model_path"])
)
if not os.path.exists(model_abs):
    st.error(f"모델 파일을 찾을 수 없습니다: `{model_abs}`")
    st.stop()

st.success(f"모델 로드 경로: `{model_abs}`")

# 이미지 업로드
uploaded_file = st.file_uploader(
    "테스트 이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col2:
        with st.spinner("모델 로딩 중..."):
            model = load_model(config)

        input_tensor = preprocess(image)

        st.subheader("예측 결과")

        if config["type"] == "multilabel":
            results = predict_multilabel(model, input_tensor, config)

            filled = [r for r in results if r["pred"]]
            empty = [r for r in results if not r["pred"]]

            st.markdown("**✅ 기입된 필드**")
            if filled:
                for r in filled:
                    st.progress(
                        r["prob"], text=f"{r['label']}  ({r['prob'] * 100:.1f}%)"
                    )
            else:
                st.info("기입된 필드 없음")

            st.markdown("**❌ 미기입 필드**")
            if empty:
                for r in empty:
                    st.progress(
                        r["prob"], text=f"{r['label']}  ({r['prob'] * 100:.1f}%)"
                    )
            else:
                st.info("미기입 필드 없음")

            total = len(results)
            n_filled = len(filled)
            st.metric(
                "기입 완성도",
                f"{n_filled}/{total} 필드",
                f"{n_filled / total * 100:.1f}%",
            )

    with col1:
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        with st.spinner("GradCAM++ 생성 중..."):
            gradcam_img = compute_gradcam_pp(model, image)
        st.image(gradcam_img, caption="GradCAM++ 히트맵", use_container_width=True)
