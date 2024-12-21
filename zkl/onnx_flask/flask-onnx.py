import numpy as np
import base64
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


w = "yolov5s.onnx"
session, output_name = onnx_load(w)
def detect(image):
    imgsz = [640, 640]
    device = torch.device("cuda:0")
    img = image.astype(numpy.float32)
    im, org_data = data_process_cv2(img, imgsz)
    y = session.run(output_name, {session.get_inputs()[0].name:im})
    pred = torch.from_numpy(y[0]).to(device)
    pred = non_max_suppression(pred)
    res_img = post_process_yolov5(pred[0], org_data)
    res_img = res_img * 255.0
    return res_img

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
            image_data = files_get.read()
            # 将原始图像二进制流转换为numpy数组格式（假设是常见图像格式能被cv2正确解析）
            img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8),  cv2.IMREAD_UNCHANGED)

            # # 对原始图像进行编码为base64字符串用于前端展示
            _, buffer_original = cv2.imencode('.jpg', img_np)
            original_image_base64 = base64.b64encode(buffer_original).decode('utf-8')

            image = detect(img_np)#加载预测图片
            _, buffer = cv2.imencode('.jpg', image)#编码为图片
            image_base64 = base64.b64encode(buffer).decode('utf-8')#编码为二进制图片

            return render_template('predict.html',user_image=original_image_base64,image=image_base64)

    except Exception as f:
        return f

# 服务器监听所有IP地址上的6008端口。
app.run(debug=True,host='0.0.0.0',port=6008)