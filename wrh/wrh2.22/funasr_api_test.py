# -*- coding: utf-8 -*-
# 调用funasr_api的接口
import requests

url = "http://127.0.0.1:5001/transcribe"

file_path = "ss.wav"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "audio/wav")}
    response = requests.post(url, files=files)

print(response.json())
