# 人工智能CV-AIGC项目-语音识别任务
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import torchaudio
import whisper

import wave
import pyaudio

app = FastAPI()

# 加载Whisper模型 (您可以选择其他模型)
model = whisper.load_model("base")  # 可以选择 "tiny", "base", "small", "medium", "large"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return content


@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    # 保存上传的音频文件
    audio_file = await file.read()
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file)

    # 加载并处理音频
    waveform, sample_rate = torchaudio.load("uploaded_audio.wav")
    audio = waveform.numpy().flatten()

    # 语音转文本
    result = model.transcribe(audio)

    return {"text": result["text"]}


@app.post("/record_audio/")
async def record_audio():
    # 录制音频并转化为文本
    audio_data = record_audio_from_mic()

    # 语音转文本
    result = model.transcribe(audio_data)

    return {"text": result["text"]}


def record_audio_from_mic():
    """从麦克风录制音频"""
    CHUNK = 1024  # 每次读取的音频块大小
    FORMAT = pyaudio.paInt16  # 音频格式
    CHANNELS = 1  # 单声道
    RATE = 16000  # 采样率
    RECORD_SECONDS = 5  # 录制时长
    WAVE_OUTPUT_FILENAME = "output/output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录制的音频保存为 .wav 文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 返回录制的音频数据
    waveform, _ = torchaudio.load(WAVE_OUTPUT_FILENAME)
    return waveform.numpy().flatten()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8060)
