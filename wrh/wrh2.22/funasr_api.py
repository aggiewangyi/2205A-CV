# -*- coding: utf-8 -*-
import os
import subprocess
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from funasr import AutoModel

# 创建 API 应用
app = FastAPI()

# 确保音频存储目录存在
UPLOAD_FOLDER = "uploads_api"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 初始化 FunASR 语音识别模型
class FunASR:
    def __init__(self) -> None:
        model_path = r"D:/Desktop/fun/asr_weight/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model_path = r"D:/Desktop/fun/asr_weight/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_model_path = r"D:/Desktop/fun/asr_weight/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

        self.model = AutoModel(
            model=model_path if os.path.exists(model_path) else "paraformer-zh",
            vad_model=vad_model_path if os.path.exists(vad_model_path) else "fsmn-vad",
            punc_model=punc_model_path if os.path.exists(punc_model_path) else "ct-punc-c",
            model_revision="master"
        )

    def transcribe(self, audio_file):
        res = self.model.generate(input=audio_file, batch_size_s=300)
        return res[0]['text']


asr = FunASR()


# 音频转换函数
def convert_to_wav(input_path):
    """使用 FFmpeg 将音频转换为 16kHz WAV 格式"""
    try:
        filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(UPLOAD_FOLDER, f"{filename_no_ext}_{timestamp}.wav")

        command = [
            "ffmpeg", "-i", input_path, "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", output_path, "-y"
        ]
        subprocess.run(command, check=True)
        return output_path
    except Exception as e:
        print("FFmpeg 转换失败:", e)
        return None


# 🎯 API 接口：上传音频并返回识别结果
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 保存上传的音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(file.filename)[-1]
        input_path = os.path.join(UPLOAD_FOLDER, f"audio_{timestamp}{file_ext}")

        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # 转换为 16kHz WAV
        converted_path = convert_to_wav(input_path)
        if not converted_path:
            return JSONResponse(content={"error": "音频格式转换失败"}, status_code=500)

        # 进行语音识别
        text = asr.transcribe(converted_path)

        return {"transcription": text, "file": file.filename}

    except Exception as e:
        return JSONResponse(content={"error": f"语音识别失败: {str(e)}"}, status_code=500)


# 运行 FastAPI 服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
# FastAPI 服务器会在 http://0.0.0.0:5001 运行
# 在浏览器打开 http://127.0.0.1:5001/docs