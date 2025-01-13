# -*- coding: utf-8 -*-
import ast
import os
import json
import cv2
import numpy as np
import onnxruntime
from flask import Flask,request
from demo01 import *
import base64
import time
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()


app = Flask(__name__,template_folder='./html')

def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    return session, output_names


#将图片转编码成二进制流
'opencv'
# def cv2_to_base64(image):
#     start_time = time.time()
#     _,image = cv2.imencode('.jpg',image)
#     end_time = (time.time()-start_time) * 1000
#     print(f'imencode编码耗时:{end_time}ms')
#     return base64.b64encode(image).decode('utf-8')

'turbojpeg'
def cv2_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoded_image_data = jpeg.encode(image_rgb, quality=75)
    return base64.b64encode(encoded_image_data).decode('utf-8')


#将二进制流解码成图片
'opencv'
# def base64_to_cv2(base64_string):
#     image_data = base64.b64decode(base64_string)
#     hu = np.frombuffer(image_data, np.uint8)
#
#     start_time = time.time()
#     image = cv2.imdecode(hu,cv2.IMREAD_COLOR)
#     end_time = (time.time() - start_time) * 1000
#
#     print(f'imdecode解码耗时:{end_time}ms')
#     return image

'turbojpeg'
def base64_to_cv2(base64_string):

    image_data = base64.b64decode(base64_string)
    start_time = time.time()
    image_rgb = jpeg.decode(image_data)
    end_time = (time.time() - start_time) * 1000
    print(f'imencode编码耗时:{end_time}ms')
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image



def data_process_cv2(img, imgsz):
    # 将图像缩放到imgsz x imgsz
    img = cv2.resize(img, imgsz)
    org_data = img.copy()
    img = np.array(img, np.float32)
    # 归一化图像
    img = img / 255.0
    image_data = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    image_data = np.ascontiguousarray(image_data)
    return image_data, org_data


# def data_process_cv2(frame, input_shape):
#     image_data, nw, nh = resize_image_cv2(frame, (input_shape[1], input_shape[0]))
#     org_data = image_data.copy()
#     np_data = np.array(image_data, np.float32)
#     np_data = np_data / 255
#     image_data = np.expand_dims(np.transpose(np_data, (2, 0, 1)), 0)
#     image_data = np.ascontiguousarray(image_data)  # 内存连续
#     return image_data, org_data
#
# def resize_image_cv2(image, size):
#     ih, iw, ic = image.shape
#     w, h = size
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#
#     image = cv2.resize(image, (nw, nh))
#     new_image = np.ones((size[0], size[1], 3), dtype='uint8') * 128
#     # new_image = np.ones((size[0], size[1], 3), dtype='uint8')
#     start_h = (h - nh) / 2
#     start_w = (w - nw) / 2
#     end_h = size[1] - start_h
#     end_w = size[0] - start_w
#     new_image[int(start_h):int(end_h), int(start_w):int(end_w)] = image
#
#     # cv2.imshow('new_image',new_image)
#     # cv2.waitKey(0)
#     return new_image, nw, nh


w = "yolov5s.onnx"
session, output_name = onnx_load(w)

label_name = session.get_modelmeta().custom_metadata_map['names']
label_name = ast.literal_eval(label_name)

device = torch.device("cuda:0")


@app.route('/post', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:

            data = json.loads(request.data)
            data = data['feed'][0]['x'].encode('utf-8')
            im0 = base64_to_cv2(data)

            im,org_data = data_process_cv2(im0,[640,640])

            y =session.run(output_name,{session.get_inputs()[0].name:im})
            pred = torch.from_numpy(y[0]).to(device)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            res_img,label= post_process_yolov5(pred[0], org_data)

            res_img = cv2_to_base64(res_img)
            res_data = {"feed": [{"img": res_img}, {"label": label}], "fetch": ["res"]}
            return res_data

        except Exception as e:
            return f"{e}"


'设置文件夹'
app.config['UPLOAD_FOLDER'] = './static/img'
'接收用户上传的图片文件'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # 设置允许的文件扩展名
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.run(host='0.0.0.0', port=2024, debug=True)

















