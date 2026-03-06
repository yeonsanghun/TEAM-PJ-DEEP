import base64
import io

import requests
from PIL import Image

import streamlit as st

# =====================================================================
# Config
# =====================================================================
API_BASE = "http://localhost:8000"
PREDICT_URL = f"{API_BASE}/efficiNetB4"

# =====================================================================
# Streamlit UI
# =====================================================================
st.set_page_config(page_title="전입신고서 분석", layout="wide")
st.title("📋 전입신고서 필드 분석")
st.caption(
    "EfficientNet-B4 모델이 전입신고서 이미지를 분석하여 14개 필드의 기입 여부를 판별합니다."
)

# ── API 서버 상태 확인 ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    api_url = st.text_input("API 서버 주소", value=API_BASE)
    predict_url = f"{api_url}/efficiNetB4"

    if st.button("서버 연결 확인"):
        try:
            resp = requests.get(f"{api_url}/", timeout=5)
            info = resp.json()
            st.success("✅ 서버 연결 성공")
            st.json(info)
        except Exception as e:
            st.error(f"❌ 서버 연결 실패: {e}")

# ── 이미지 업로드 ───────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "전입신고서 이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    # ── API 호출 ────────────────────────────────────────────────────
    with st.spinner("서버에서 분석 중..."):
        uploaded_file.seek(0)
        try:
            resp = requests.post(
                predict_url,
                files={
                    "file": (uploaded_file.name, uploaded_file.read(), "image/jpeg")
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ API 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            st.stop()

    predictions = data["predictions"]
    gradcam_b64 = data["gradcam_b64"]
    pred_class_name = data["pred_class_name"]
    n_filled = data["n_filled"]
    total = data["total"]

    # GradCAM 이미지 디코딩
    gradcam_bytes = base64.b64decode(gradcam_b64)
    gradcam_img = Image.open(io.BytesIO(gradcam_bytes))

    # ── 좌측: 원본 이미지 + GradCAM ────────────────────────────────
    with col1:
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        st.image(
            gradcam_img,
            caption=f"GradCAM++ 히트맵 (주목 클래스: {pred_class_name})",
            use_container_width=True,
        )

    # ── 우측: 예측 결과 ─────────────────────────────────────────────
    with col2:
        st.subheader("예측 결과")

        filled = [r for r in predictions if r["pred"]]
        empty = [r for r in predictions if not r["pred"]]

        st.markdown("**✅ 기입된 필드**")
        if filled:
            for r in filled:
                st.progress(r["prob"], text=f"{r['label']}  ({r['prob'] * 100:.1f}%)")
        else:
            st.info("기입된 필드 없음")

        st.markdown("**❌ 미기입 필드**")
        if empty:
            for r in empty:
                st.progress(r["prob"], text=f"{r['label']}  ({r['prob'] * 100:.1f}%)")
        else:
            st.info("미기입 필드 없음")

        st.metric(
            "기입 완성도",
            f"{n_filled}/{total} 필드",
            f"{n_filled / total * 100:.1f}%",
        )
