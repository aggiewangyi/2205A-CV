
from flask import Flask, render_template, request, jsonify# 网站框架
import os
import torch
from funasr import AutoModel# 语音识别模型
import soundfile as sf
import numpy as np
import io
import pyaudio# 录音功能
import wave
from datetime import datetime
# 初始化网站和语音识别模型
app = Flask(__name__)

# 初始化FunASR模型
model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1") # 加载中文语音识别模型


# 录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10

# 主页
@app.route('/')
def index():
    return render_template('index.html')# 返回网页界面

# 处理文件上传
@app.route('/upload', methods=['POST'])
def upload_file():
    # 处理文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file'] # 拿到用户上传的文件
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    # 保存并处理音频文件
    audio_data, sr = sf.read(io.BytesIO(file.read()))# 将文件转为模型能理解的格式
    text = model.generate(audio_data)#语音识别转换成文本
    return jsonify({'result': text[0]['text']})# 返回识别结果


@app.route('/record', methods=['POST'])
def record_audio():
    # 实时录音处理
    p = pyaudio.PyAudio() # 使用pyaudio库
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 转换为numpy数组
    audio_np = np.frombuffer(b''.join(frames), dtype=np.int16)#转换为数字信号
    text = model.generate(audio_np)
    return jsonify({'result': text[0]['text']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)