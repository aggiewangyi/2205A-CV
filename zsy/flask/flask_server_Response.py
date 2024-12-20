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
    file = request.files['image']
    image_str = file.read()
    nparr = np.frombuffer(image_str, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dected_image, name_list = detect_object(frame, session, model_inputs, input_width, input_height)

    # cv2.imshow("img_base64_encoded", img_base64_encoded)
    # cv2.waitKey(0)

    buffer = cv2.imencode('.jpg', dected_image)[1]

    res_bytes = buffer.tobytes()
    # base64_str = base64.b64encode(buffer).decode('utf-8')

    # name_list = name_list.tolist()
    #
    # res_data = {"feed": [{"img": base64_str}, {"label": name_list}], "fetch": ["res"]}
    return Response(res_bytes, mimetype='image/jpeg')
    # return render_template("predict.html", base64_image=base64_str)

   # return
    # return render_template("predict.html", base64_str=encoded_image)
    # red_json = json.dumps(res_data)

    #return res_data

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=97)








