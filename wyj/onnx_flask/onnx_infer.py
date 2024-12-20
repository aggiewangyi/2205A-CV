import ast
import time
import cv2
import numba
import onnx
import onnxruntime as rt
import torch
import os
from tool import *
import numpy as np

from flask import Flask,render_template,request
app = Flask(__name__)

def onnx_load(model):
    session = rt.InferenceSession(model,providers='CPU')
    input_names = session.get_inputs()[0].name
    output_names = session.get_outputs()[0].name
    return session,input_names,output_names

model = 'yolov5s.onnx'
session,input_names,output_names = onnx_load(model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['post'])
def predict():
    file = request.files['file']
    try:
        if file:
            img_path = f'./static/images/{file.filename}'
            file.save(img_path)
            img_dir = './static/images'
            img_size = [640,640]

            ori_img = cv2.imdecode(np.fromfile(img_path,dtype='uint8'),cv2.IMREAD_UNCHANGED)
            if ori_img.shape[2] >3:
                ori_img = cv2.cvtColor(ori_img,cv2.COLOR_RGBA2RGB)
            img,img0 = data_process_cv2(ori_img,img_size)
            # start_time = time.time()
            res = session.run([output_names],{input_names:img})
            pred = torch.from_numpy(res[0]).cpu()
            pred = non_max_suppression(pred,conf_thres=0.5,iou_thres=0.5,max_det=1000)
            # print("spend time:{0} ms".format(time.time() - start_time))
            label_names = eval(session.get_modelmeta().custom_metadata_map['names'])
            res_img = post_process_yolov5(pred[0], ori_img ,img.shape[2:],label_names)
            res_path = os.path.join(img_dir,'res.jpg')
            cv2.imwrite(res_path,res_img)
            return render_template('predict.html',user_image=res_path)
    except Exception as e:
        print(e)
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)