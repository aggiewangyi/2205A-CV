import base64
import json

import cv2
from flask import Flask, request, render_template, Response
from tool import *

app = Flask(__name__)

model_path = "yolov8n.onnx"
session, model_inputs, input_width, input_height = init_detect_model(model_path)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/post', methods=['POST'])
def post_example():
    print('get into server!')
    data = json.loads(request.data)
    data = base64.b64decode(data['feed'][0]["x"].encode('utf8'))
    data = np.frombuffer(data, np.uint8)

    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    dected_image, name_list = detect_object(frame, session, model_inputs, input_width, input_height)

    # cv2.imshow("img_base64_encoded", dected_image)
    # cv2.waitKey(0)


    buffer = cv2.imencode('.jpg', dected_image)[1]
    base64_str = base64.b64encode(buffer).decode('utf-8')
    if name_list == "":
        name_list = None
    else:
        name_list = name_list.tolist()

    res_data = {"feed": [{"img": base64_str}, {"label": name_list}], "fetch": ["res"]}

    red_json = json.dumps(res_data)

    return red_json

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=97)








