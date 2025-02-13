# 人工智能 CV-AIGC 项目-语音识别任务
from flask import Flask, request, jsonify, render_template
from FunASR import FunASR

app = Flask(__name__, template_folder='.')



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', recognized_text='')


@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            print(file)
            text = model.transcribe(file)
            # print(text)
            return text

        except:
            return '错误',500




if __name__ == '__main__':
    model = FunASR()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=6008, debug=True)

