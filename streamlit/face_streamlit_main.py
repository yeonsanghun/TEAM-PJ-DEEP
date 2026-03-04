import requests
from PIL import Image

import streamlit as st

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="얼굴 분류 데모",
    page_icon="🙂",
    layout="centered",
)

FASTAPI_URL = "http://localhost:7394/face_infer"
CLASS_NAMES = ["마동석", "카리나", "장원영"]
CLASS_EMOJI = ["💪", "🌟", "🌸"]

# ── UI ───────────────────────────────────────────────────────
st.title("🙂 얼굴 분류 데모")
st.caption("ResNet34 모델이 마동석 · 카리나 · 장원영 중 누구인지 판별합니다.")
st.divider()

uploaded = st.file_uploader(
    "이미지를 업로드하세요 (jpg / jpeg / png / webp)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    col_img, col_result = st.columns([1, 1], gap="large")

    # 왼쪽: 업로드된 이미지 미리보기
    with col_img:
        st.subheader("업로드된 이미지")
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

    # 오른쪽: 추론 결과
    with col_result:
        st.subheader("추론 결과")
        with st.spinner("모델 추론 중..."):
            try:
                # multipart/form-data 로 FastAPI 서버에 전송
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(FASTAPI_URL, files=files, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    st.error(f"서버 에러: {data['error']}")
                else:
                    name = data.get("result", "알 수 없음")
                    idx = data.get("index", -1)
                    fname = data.get("filename", "")

                    emoji = CLASS_EMOJI[idx] if 0 <= idx < len(CLASS_EMOJI) else "❓"

                    st.success("추론 완료!")
                    st.markdown(
                        f"""
                        <div style='text-align:center;padding:24px;
                             border-radius:12px;background:#f0f2f6;'>
                          <div style='font-size:64px;'>{emoji}</div>
                          <div style='font-size:36px;font-weight:bold;
                               margin-top:8px;'>{name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.divider()
                    st.markdown("**상세 정보**")
                    st.json({"result": name, "index": idx, "filename": fname})

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI 서버에 연결할 수 없습니다.\n\n"
                    "`uvicorn fastapi_main:app --port 7394` 가 실행 중인지 확인하세요."
                )
            except requests.exceptions.Timeout:
                st.error("서버 응답 시간이 초과되었습니다. 다시 시도해 주세요.")
            except Exception as e:
                st.error(f"오류 발생: {e}")
