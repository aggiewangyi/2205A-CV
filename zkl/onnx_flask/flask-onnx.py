# -*- coding: utf-8 -*-

import numpy as np

from flask import Flask, render_template, request
import random, onnxruntime

import torch.cuda
from demo01 import *
random.seed(0)

app = Flask(__name__,template_folder='./html')

def data_process_cv2(img, imgsz):
    # 将图像缩放到imgsz x imgsz
    img = cv2.resize(img, (imgsz))
    # 归一化图像
    img = img / 255.0
    input_array = np.expand_dims(img, axis=0)
    input_array = np.transpose(input_array, (0, 3, 1, 2))

    return input_array, img

def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_name = [x.name for x in session.get_outputs()]
    return session, output_name

def detect(path):
    w = "yolov5s.onnx"
    imgsz = [640, 640]
    session, output_name = onnx_load(w)
    device = torch.device("cuda:0")
    img = cv2.imread(path)
    img = img.astype(numpy.float32)
    im, org_data = data_process_cv2(img, imgsz)
    y = session.run(output_name, {session.get_inputs()[0].name:im})
    pred = torch.from_numpy(y[0]).to(device)
    pred = non_max_suppression(pred)


    res_img = post_process_yolov5(pred[0], org_data)
    res_img = res_img * 255.0
    return res_img
    # cv2.imwrite('image.jpg',res_img)
    # cv2.imshow('res', res_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#
# image_detect = detect(r"C:\Users\26296\Desktop\v5\datasets\coco128\images\train2017\000000000009.jpg")  # 加载预测图片
# cv2.imwrite('image.jpg',image_detect)

'渲染home网页模板'
@app.route('/')
def home():
    return render_template('home.html')

'进行图片预测'
@app.route('/predict',methods=['POST'])
def predict():
    files_get = request.files.get('file')
    try:
        if files_get:
            filename = files_get.filename#获取文件名
            path = os.path.join('./static/images',filename)#文件路径
            files_get.save(path)#将文件保存进指定路径
            print(path)

            image = detect(path)#加载预测图片
            cv2.imwrite('C:/Users/26296/Desktop/onnx_flask/static/images/image.jpg',image)
            return render_template('predict.html',user_image=path,image='./static/images\image.jpg')

    except Exception as f:
        return f




# 服务器监听所有IP地址上的6008端口。
app.run(debug=True,host='0.0.0.0',port=6008)