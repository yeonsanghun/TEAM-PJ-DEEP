import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

import streamlit as st

# 클래스 이름 정의
class_names = ["마동석", "카리나", "이수지"]


# 모델 불러오기 함수
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(
        torch.load("../best_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


# 이미지 전처리 함수
def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # (1, C, H, W)


# Streamlit UI
st.title("🎬 연예인 분류기 (마동석 / 카리나 / 이수지)")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    model = load_model()
    input_tensor = transform_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
        confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100

    st.success(f"예측 결과: **{pred_class}** ({confidence:.2f}% 확신)")
