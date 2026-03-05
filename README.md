fastapi 실행
cd c:\team2\TEAM-PJ-DEEP\fastapi ; ..\\.venv\Scripts\uvicorn.exe fastapi_main:app --reload --host 127.0.0.1 --port 7394

swagger
http://localhost:7394/docs

streamlit 실행
.venv\Scripts\streamlit.exe run streamlit\face_streamlit_main.py --server.port 8501

http://localhost:8501/

멀티라벨 샘플코드
https://www.notion.so/CNN-319179cfab9480bf98e5de00ecdf30c8?source=copy_link

모델학습
cd "c:\team2\TEAM-PJ-DEEP\multi_label" ; & "c:\team2\TEAM-PJ-DEEP\.venv\Scripts\python.exe" multi_label_main.py

멀티라벨 테스트를 위한 fastapi server 실행
cd "c:\team2\TEAM-PJ-DEEP\multi_label" ; & "c:\team2\TEAM-PJ-DEEP\.venv\Scripts\uvicorn.exe" multi_lable_fastapi:app --host 0.0.0.0 --port 7394

whisper
TEAM-PJ-DEEP> .\.venv\Scripts\python.exe stt\whisper_main.py Leejamsample.mp3
