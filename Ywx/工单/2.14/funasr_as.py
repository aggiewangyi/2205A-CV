from flask import Flask, request, jsonify
from funasr import AutoModel

app = Flask(__name__)

# 加载模型
model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    disable_update=True
)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        audio_file = request.files['file']  # 获取上传的文件
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)  # 保存文件

        # 进行语音识别
        res = model.generate(input=audio_path, batch_size_s=300)
        return jsonify({"text": res[0]['text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080, debug=True)  # 监听本地 5000 端口
