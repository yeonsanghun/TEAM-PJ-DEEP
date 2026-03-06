import os
import sys

import whisper

if len(sys.argv) < 2:
    print("Usage: python whisper_main.py <audio_filename>")
    sys.exit(1)

model = whisper.load_model("turbo")
input_audio_dir = "audio"
input_audio_file = sys.argv[1]
input_audio = whisper.load_audio(os.path.join(input_audio_dir, input_audio_file))
input_audio_name = input_audio_file.split("/")[-1]

audio_path = os.path.join(
    os.path.dirname(__file__), "..", input_audio_dir, input_audio_name
)
result = model.transcribe(audio_path)
print(result["text"])

# sttout 디렉토리에 결과 저장
base_name = os.path.splitext(input_audio_file)[0]
output_dir = os.path.join(os.path.dirname(__file__), "..", "sttout")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{base_name}.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result["text"])
print(f"저장 완료: {output_path}")
