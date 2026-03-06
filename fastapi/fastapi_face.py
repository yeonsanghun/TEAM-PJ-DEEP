# ============================================================
# FastAPI 기반 이미지 분류 추론 서버
# - ResNet34 모델을 사용하여 업로드된 이미지를 분류합니다.
# - 클래스:
# - 업로드된 이미지는 upload_img/ 폴더에 저장됩니다.
# await file.read()          → bytes    (압축된 원시 데이터)
#     ↓ Image.open(BytesIO(img)).convert("RGB")
# PIL Image                  → 픽셀 배열 객체 (H × W × 3)
#     ↓ transform_test(img_data)
# torch.Tensor               → (3, 224, 224) 형태의 숫자 배열
# ============================================================
#
# ┌─────────────────────────────────────────────────────────┐
# │              전체 실행 흐름 (Execution Flow)              │
# │                                                         │
# │  [서버 기동 시]                                           │
# │   Step 1. 라이브러리 임포트 및 전역 디바이스 설정           │
# │   Step 2. lifespan — 모델 로드 (Startup)                 │
# │   Step 3. 이미지 전처리 파이프라인 정의                    │
# │                                                         │
# │  [요청 수신 시 — POST /infer 호출 순서]                    │
# │   Step 4. 파일 수신 → 확장자 검증 → 디스크 저장            │
# │   Step 5. bytes → PIL Image 변환                        │
# │   Step 6. PIL Image → torch.Tensor 변환                 │
# │   Step 7. torch.Tensor → 클래스 인덱스 (모델 추론)         │
# │   Step 8. 클래스 인덱스 → 최종 JSON 응답 생성             │
# │                                                         │
# │  [서버 종료 시]                                           │
# │   Step 2. lifespan — 자원 해제 (Shutdown)               │
# └─────────────────────────────────────────────────────────┘


# ============================================================
# ## Step 1. 라이브러리 임포트 및 전역 디바이스 설정
#
# 서버가 처음 실행될 때 Python 인터프리터가 모듈을 적재하면서
# 이 블록이 가장 먼저 평가됩니다.
# - 표준 라이브러리(io, os, uuid)와 서드파티 패키지(torch, fastapi 등)를 불러옵니다.
# - device 변수는 이후 모든 텐서·모델 연산에서 공통으로 참조합니다.
# ============================================================

import io  # 바이트 데이터를 파일처럼 다루기 위한 표준 라이브러리
import os  # 파일 경로 처리, 디렉토리 조작을 위한 표준 라이브러리
import tempfile  # 임시 파일 생성
import uuid  # 업로드 파일에 고유한 이름을 부여하기 위한 라이브러리
from contextlib import asynccontextmanager  # 비동기 컨텍스트 매니저 데코레이터

import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 레이어 정의 (Linear 등)
import torchvision.models as models  # 사전 학습된 모델 불러오기 (ResNet 등)
import torchvision.transforms as transforms  # 이미지 전처리 파이프라인
import whisper  # OpenAI Whisper STT 모델
from PIL import Image  # 이미지 열기 및 변환을 위한 Pillow 라이브러리

from fastapi import (  # FastAPI 웹 프레임워크 및 파일 업로드 관련 클래스
    FastAPI,
    File,
    Request,
    UploadFile,
)

# 사용 가능한 경우 GPU(cuda)를 사용하고, 그렇지 않으면 CPU를 사용
# - 이 값은 모델 로드(Step 2), 텐서 변환(Step 6), 추론(Step 7)에서 공통 참조됩니다.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ## Step 2. 서버 생명주기 관리 — 모델 로드(Startup) / 자원 해제(Shutdown)
#
# FastAPI의 lifespan 패턴을 사용해 서버 기동·종료 시점의 작업을 한 함수에서 관리합니다.
#
# [Startup — yield 이전]
#   1) ResNet34 구조 생성 및 분류기(fc 레이어) 교체
#   2) 체크포인트 파일에서 학습된 가중치 로드
#   3) 모델을 추론 모드(eval)로 전환 후 app.state에 저장
#
# [Shutdown — yield 이후]
#   - 서버 종료 메시지 출력 (필요 시 DB 연결 해제 등 정리 작업 추가)
#
# ※ 전역 변수 대신 app.state를 사용하는 이유
#   - 의존성이 명확해져 테스트 시 모델을 쉽게 교체할 수 있습니다.
#   - 엔드포인트에서 request.app.state.model 로 어디서든 접근 가능합니다.
# ============================================================


# @asynccontextmanager
# - contextlib 모듈의 데코레이터로, 일반 비동기 제너레이터 함수를
#   'async with' 구문에서 사용할 수 있는 비동기 컨텍스트 매니저로 변환합니다.
# - yield 이전 코드 → 서버 시작(startup) 시 실행
# - yield 이후 코드 → 서버 종료(shutdown) 시 실행
# - FastAPI의 lifespan 매개변수에 전달하여 앱 생명주기를 관리합니다.
@asynccontextmanager
# async def lifespan(app: FastAPI):
# - FastAPI 인스턴스(app)를 인자로 받는 비동기 생명주기 함수입니다.
# - 서버 기동 ~ 종료까지의 전체 생명주기(lifespan)를 하나의 함수로 정의합니다.
# - app.state에 모델 등 공유 자원을 저장하고, yield 뒤에서 정리 작업을 수행합니다.
# - 기존 @app.on_event("startup") / @app.on_event("shutdown") 방식을 대체합니다.
async def lifespan(app: FastAPI):
    print("========== 모델 불러오기 시작 ==============")
    # global 대신 app.state에 모델을 저장합니다.
    # app.state는 FastAPI 앱 인스턴스에 속한 딕셔너리로,
    # 서버가 살아있는 동안 유지되며 어디서든 request.app.state로 접근할 수 있습니다.

    # ResNet34 모델 구조를 불러옵니다 (pretrained=True: ImageNet 사전 학습 가중치 포함)
    _model = models.resnet34(pretrained=True)

    # ResNet34의 기본 출력 클래스(1000개)를 우리 데이터셋 클래스 수(3개)로 교체
    # fc(Fully Connected) 레이어: 입력 512차원 → 출력 3차원
    _model.fc = nn.Linear(512, 3)

    # 저장된 체크포인트 파일을 불러옵니다.
    # map_location=device: 저장 당시 GPU였어도 현재 환경(CPU/GPU)에 맞게 로드
    checkpoint = torch.load("../best_model/best_model_face.pth", map_location=device)

    # 체크포인트 저장 방식에 따라 state_dict 추출 방법이 다릅니다.
    # - 딕셔너리 형태로 저장된 경우: checkpoint["model_state_dict"] 키로 접근
    # - state_dict만 바로 저장된 경우: checkpoint 자체가 state_dict
    if "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    else:
        _model.load_state_dict(checkpoint)

    print(device)  # 현재 사용 중인 디바이스 출력 (cuda 또는 cpu)
    _model.to(device)  # 모델을 해당 디바이스(GPU/CPU)로 이동
    _model.eval()  # 평가 모드로 전환: Dropout, BatchNorm 등이 추론에 맞게 동작

    # 전역 변수 대신 app.state에 저장 → request.app.state.model 로 접근
    app.state.model = _model
    print("========== 모델 불러오기 끝!!! ==============")

    # Whisper STT 모델 로드
    print("========== Whisper 모델 불러오기 시작 ===========")
    app.state.whisper_model = whisper.load_model("turbo")
    print("========== Whisper 모델 불러오기 끝!!! ==========")

    yield  # ← 이 지점에서 실제 서버가 요청을 처리하기 시작합니다.

    # [Shutdown] yield 이후 코드는 서버가 종료될 때 실행됩니다.
    print("서버 종료.")


# ============================================================
# FastAPI 앱 인스턴스 생성
# - lifespan=lifespan: Step 2에서 정의한 생명주기 함수를 앱에 등록합니다.
# - app.state: 앱 전체에서 공유할 객체(모델 등)를 저장하는 공간
#   전역 변수 대신 app.state를 사용하면 의존성이 명확해지고
#   테스트 시 모델을 쉽게 교체할 수 있습니다.
# ============================================================
app = FastAPI(lifespan=lifespan)


# ============================================================
# ## Step 3. 이미지 전처리 파이프라인 정의
#
# 서버 기동 시 모듈 수준에서 한 번만 생성되며, 요청마다 재사용됩니다.
# 업로드된 이미지를 ResNet34 모델이 요구하는 입력 형식으로 변환합니다.
#
# 변환 순서:
#   Resize(224×224) → ToTensor([0,1] FloatTensor) → Normalize(ImageNet 통계값)
#
# ResNet 계열 모델은 ImageNet으로 사전 학습되었으므로,
# 동일한 평균(mean)·표준편차(std)로 정규화해야 피처 분포가 일치합니다.
# ============================================================
transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 리사이즈
        transforms.ToTensor(),  # PIL 이미지 → [0,1] 범위 FloatTensor (C, H, W)
        # ImageNet 데이터셋의 평균(mean)과 표준편차(std)로 정규화
        # mean: [0.485, 0.456, 0.406] (R, G, B 채널별)
        # std:  [0.229, 0.224, 0.225] (R, G, B 채널별)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ============================================================
# ## Step 4. 파일 수신 → 확장자 검증 → 디스크 저장
#
# POST /infer 엔드포인트에서 가장 먼저 호출되는 전처리 단계입니다.
# 보안상 허용된 이미지 확장자만 수락하고, 유효한 파일은 upload_img/ 에 저장합니다.
#
# 반환값: (저장된 파일명, 에러딕셔너리) 튜플
#   - 성공 시: ("uuid.ext", None)
#   - 실패 시: (None, {"error": "..."})
#
# 튜플 반환을 사용하는 이유:
#   - 예외(Exception)를 던지면 FastAPI가 500 에러를 반환하지만,
#     에러 딕셔너리를 반환하면 호출부에서 직접 JSON 응답으로 전달할 수 있어
#     클라이언트 친화적인 에러 메시지를 전달하기 쉽습니다.
# ============================================================
def validate_and_save(img_bytes: bytes, filename: str):
    # 허용되는 이미지 확장자 목록
    allowed_ext = ["jpg", "jpeg", "png", "webp"]

    # 업로드된 파일명에서 확장자를 추출합니다.
    # 예: "photo.PNG" → split('.') → ['photo', 'PNG'] → [-1].lower() → 'png'
    ext = filename.split(".")[-1].lower()

    # 허용되지 않는 확장자인 경우 에러 딕셔너리를 반환합니다.
    if ext not in allowed_ext:
        return None, {"error": "이미지파일만 업로드하세요!!!!!"}

    # uuid4()로 고유한 파일명을 생성하여 중복 저장 문제를 방지합니다.
    # 예: "3f2504e0-4f89-11d3-9a0c-0305e82c3301.jpg"
    newfile_name = f"{uuid.uuid4()}.{ext}"

    # upload_img 디렉토리에 저장할 전체 파일 경로를 생성합니다.
    file_path = os.path.join("../upload_img", newfile_name)

    # 이미지 파일을 바이너리 쓰기 모드("wb")로 upload_img 폴더에 저장합니다.
    with open(file_path, "wb") as buffer:
        buffer.write(img_bytes)

    return newfile_name, None  # 성공 시 에러 없음


# ============================================================
# ## Step 5. bytes → PIL Image 변환
#
# 네트워크를 통해 수신된 원시 바이트(bytes)를 픽셀 단위로 조작 가능한
# PIL Image 객체로 변환합니다.
#
# - io.BytesIO : 바이트 데이터를 디스크에 저장하지 않고 메모리 상에서
#                파일처럼 읽을 수 있게 해주는 버퍼입니다.
# - .convert("RGB") : PNG(RGBA), 흑백(L) 등 다양한 포맷을 3채널 RGB로 통일합니다.
#                     모델은 항상 3채널 입력을 기대하기 때문에 필수 변환입니다.
# ============================================================
def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(pil_img.size)  # 원본 이미지 크기 확인용 출력 (width, height)
    return pil_img


# ============================================================
# ## Step 6. PIL Image → torch.Tensor 변환
#
# Step 3에서 정의한 transform_test 파이프라인을 적용해
# PIL Image를 모델이 받아들일 수 있는 torch.Tensor로 변환합니다.
#
# 변환 결과 shape 변화:
#   PIL Image (H, W, 3)
#     → ToTensor  → (3, H, W)         [0.0 ~ 1.0 FloatTensor]
#     → Normalize → (3, H, W)         [정규화된 값]
#     → Resize    → (3, 224, 224)
#     → unsqueeze → (1, 3, 224, 224)  [배치 차원 추가]
#     → .to(device) → GPU 또는 CPU 텐서
# ============================================================
def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    # (3, 224, 224) → (1, 3, 224, 224)
    input_tensor = transform_test(pil_img).unsqueeze(0).to(device)
    return input_tensor


# ============================================================
# ## Step 7. torch.Tensor → 클래스 인덱스 (모델 추론)
#
# 전처리된 텐서를 모델에 통과시켜 각 클래스의 로짓(logit) 점수를 계산하고,
# 가장 높은 점수의 클래스 인덱스를 반환합니다.
#
# - model(input_tensor) : 순전파(forward pass) 수행 → shape (1, 3)
# - torch.argmax(dim=1) : 배치 내 각 샘플에서 최대값 인덱스 반환 → shape (1,)
# - .item()             : 텐서 스칼라를 Python int로 변환 → 0, 1, 또는 2
#
# model을 전역 변수가 아닌 인자로 받는 이유:
#   의존성이 명확해져 단위 테스트 시 mock 모델로 쉽게 교체할 수 있습니다.
# ============================================================
def predict(model: nn.Module, input_tensor: torch.Tensor) -> int:
    # model을 전역 변수 대신 인자로 받아 의존성을 명확히 합니다.
    pred = model(input_tensor)  # pred shape: (1, 3)
    class_idx = torch.argmax(pred, dim=1).item()  # 0, 1, 2 중 하나
    return class_idx


# ============================================================
# ## Step 8. 클래스 인덱스 → 최종 JSON 응답 생성
#
# 추론 결과인 정수 인덱스(0~2)를 사람이 읽을 수 있는 클래스 이름으로 변환하고,
# 클라이언트에 반환할 JSON 응답 딕셔너리를 조립합니다.
#
# 반환값 구조:
#   {
#     "result"  : "마동석" | "카리나" | "장원영",  # 예측된 클래스 이름
#     "index"   : 0 | 1 | 2,                      # 원본 클래스 인덱스
#     "filename": "uuid.ext"                       # 서버에 저장된 파일명
#   }
# ============================================================
def make_response(class_idx: int, newfile_name: str) -> dict:
    # 인덱스(0, 1, 2)에 대응하는 클래스 이름 목록
    model_class = ["마동석", "카리나", "장원영"]
    return {
        "result": model_class[class_idx],  # 인덱스 → 사람 이름으로 변환
        "index": class_idx,  # 원본 인덱스도 함께 반환
        "filename": newfile_name,  # 서버에 저장된 파일명
    }


# ============================================================
# ## 라우터 정의 (1) — GET /  서버 상태 확인
#
# 서버가 정상적으로 기동·응답 중인지 확인하는 헬스체크 엔드포인트입니다.
# 브라우저나 curl로 접근해 간단히 동작 여부를 파악할 수 있습니다.
# ============================================================
@app.get("/")
def root():
    # 서버가 정상 동작 중임을 알려주는 간단한 응답 반환
    return {"result": "Hi!!!"}


# ============================================================
# ## 라우터 정의 (2) — POST /infer  이미지 분류 추론
#
# 클라이언트가 multipart/form-data 형식으로 이미지 파일을 업로드하면
# Step 4 ~ Step 8을 순서대로 호출하여 분류 결과를 JSON으로 반환합니다.
#
# 요청 형식: multipart/form-data  (file 필드에 이미지 첨부)
# 응답 형식: JSON { result, index, filename }
#
# 전체 처리 흐름:
#   [파일 수신] await file.read()
#       → [Step 4] validate_and_save()   확장자 검증 + 디스크 저장
#       → [Step 5] bytes_to_pil()        bytes → PIL Image
#       → [Step 6] pil_to_tensor()       PIL Image → torch.Tensor
#       → [Step 7] predict()             Tensor → 클래스 인덱스
#       → [Step 8] make_response()       인덱스 → JSON 응답
# ============================================================
@app.post("/face_infer")
async def face_infer(
    request: Request, file: UploadFile = File(...)
):  # body : form-data
    # request.app.state.model: startup 시 저장한 모델을 전역 변수 없이 꺼냅니다.

    # 업로드된 파일의 내용을 바이트 형태로 읽어옵니다.
    img = await file.read()

    # Step 4: 확장자 검증 + 원본 파일 저장
    # - 허용되지 않는 확장자면 error 딕셔너리를 반환하고 즉시 종료
    # - 성공 시 uuid 기반 고유 파일명으로 upload_img/ 에 저장
    newfile_name, error = validate_and_save(img, file.filename)
    if error:
        return error

    # -----------------추론 코드 (Step 5 ~ Step 8)--------------------------

    # Step 5: bytes → PIL Image (압축 해제, RGB 변환)
    pil_img = bytes_to_pil(img)

    # Step 6: PIL Image → torch.Tensor (리사이즈, 정규화, 배치 차원 추가)
    input_tensor = pil_to_tensor(pil_img)

    # Step 7: torch.Tensor → 클래스 인덱스 (모델 추론)
    # app.state에서 모델을 꺼내 predict()에 직접 전달합니다.
    result = predict(request.app.state.model, input_tensor)

    # Step 8: 클래스 인덱스 + 파일명 → 최종 JSON 응답 생성
    return make_response(result, newfile_name)


# ============================================================
# ## 라우터 정의 (3) — POST /stt  음성 파일 → 텍스트 변환
#
# 클라이언트가 mp3 등 오디오 파일을 업로드하면 Whisper 모델로 전사(transcription)하여
# 텍스트를 JSON으로 반환합니다.
#
# 요청 형식: multipart/form-data  (file 필드에 오디오 파일 첨부)
# 응답 형식: JSON { filename, text }
# ============================================================
@app.post("/stt")
async def stt(request: Request, file: UploadFile = File(...)):
    allowed_ext = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_ext:
        return {"error": f"지원하지 않는 형식입니다: .{ext}"}

    audio_bytes = await file.read()

    # 업로드된 바이트를 임시 파일로 저장 후 Whisper에 전달
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = request.app.state.whisper_model.transcribe(tmp_path)
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "text": result["text"]}


# ============================================================
# ## 서버 직접 실행 진입점
#
# `python fastapi_main.py` 로 직접 실행할 때 uvicorn 서버를 기동합니다.
# (VS Code 디버거 또는 터미널에서 직접 실행 시 사용)
# ============================================================
if __name__ == "__main__":
    import uvicorn

    # ※ 브레이크포인트 디버깅 시 반드시 reload=False 로 설정해야 합니다.
    #   reload=True 이면 uvicorn이 자식 프로세스를 생성하여
    #   VS Code 디버거가 자식 프로세스에 연결되지 않아 브레이크포인트가 무시됩니다.
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=7394,
        reload=False,
        log_level="debug",
    )
