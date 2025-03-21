# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import random
import onnxruntime
from tool import *
import time
import ast


def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    return session, output_names



from flask import Flask, request, render_template
import os
import json
import cv2 as cv

# 网页api
app = Flask(__name__, template_folder='.')
w = "yolov5l6.onnx"
# image dir = "images"
imgsz = [640, 640]
session, output_names = onnx_load(w)
# print(output_names)
label_name = session.get_modelmeta().custom_metadata_map['names']
# print(session.get_modelmeta())
# label_name = eval(label_name)
label_name = ast.literal_eval(label_name)
# print(label_name)
device = torch.device("cuda:0")
# 提供一个简单的用户界面，允许用户上传图片。
# @app.route('/')
# def home():
#     return render_template('./templates/home.html')
import base64


# 将图片转编码成二进制流
def cv2_to_base64(image):
    image = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(image).decode('utf-8')


# 将二进制流解码成图片
def base64_to_cv2(base64_string):
    image_data = base64.b64decode(base64_string)
    data = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image


@app.route('/post', methods=['POST', 'GET'])
def pre():
    # print(request.method)
    if request.method == 'POST':
        try:
            data = json.loads(request.data)
            data = data['feed'][0]['x'].encode('utf8')
            im0 = base64_to_cv2(data)

            # im0原图
            im, org_data = data_process_cv2(im0, imgsz)
            # im:[1,3,640,640]变换维度后  org_data:[640,640,3]变换维度前
            y = session.run(output_names, {session.get_inputs()[0].name: im})
            pred = torch.from_numpy(y[0]).to(device)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            res_img, name_list = post_process_yolov5(pred[0], im0, label_name, org_data)
            print(name_list)

        except Exception as f:
            return f'{f}'
    res_img = cv2_to_base64(res_img)
    res_data = {"feed": [{"img": res_img}, {"label": name_list}], "fetch": ["res"]}
    return res_data


# 设置上传文件夹
app.config['UPLOAD_FOLDER'] = './static/img'
# 接收用户上传的图片文件。
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # 设置允许的文件扩展名
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.run(host='0.0.0.0', port=6008, debug=True)
#
