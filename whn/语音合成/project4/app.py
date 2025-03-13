# 人工智能CV-AIGC 项目-语音合成任务
from flask import Flask, render_template, request, send_file
from tts_service import generate_speech

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = request.form["text"]
    output_path = "static/output.wav"

    # 调用本地语音合成服务
    generate_speech(text, output_path)

    return {"audio_url": "/static/output.wav"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)