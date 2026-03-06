fastapi 실행
cd c:\team2\TEAM-PJ-DEEP\fastapi ; ..\\.venv\Scripts\uvicorn.exe fastapi_main:app --reload --host 127.0.0.1 --port 7394

swagger
http://localhost:7394/docs

..\.venv\Scripts\streamlit.exe run model_tester.py --server.port 8501

http://localhost:8501/

멀티라벨 샘플코드
https://www.notion.so/CNN-319179cfab9480bf98e5de00ecdf30c8?source=copy_link

모델학습
cd efficiNetB4
..\.venv\Scripts\python.exe .\efficiNetB4_main.py

멀티라벨 테스트를 위한 fastapi server 실행
cd "c:\team2\TEAM-PJ-DEEP\multi_label" ; & "c:\team2\TEAM-PJ-DEEP\.venv\Scripts\uvicorn.exe" multi_lable_fastapi:app --host localhost --port 7394

whisper
TEAM-PJ-DEEP> .\.venv\Scripts\python.exe stt\whisper_main.py Leejamsample.mp3

streamlit
streamlit 실행
cd streamlit
..\.venv\Scripts\streamlit.exe run model_tester.py --server.port 8501
Local URL: http://localhost:8501
Network URL: http://172.16.20.248:8501
