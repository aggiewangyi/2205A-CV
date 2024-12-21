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


# if __name__ == "__main__":
#     # w = "yolov5s.onnx"
#     w = "yolov5l6.onnx"
#     # image dir = "images"
#     image_dir = r"C:\Users\quant\Desktop\cs\coco128\images\train2017"
#     imgsz = [640, 640]
#     session, output_names = onnx_load(w)
#     # print(output_names)
#     label_name = session.get_modelmeta().custom_metadata_map['names']
#     print(session.get_modelmeta())
#     # label_name = eval(label_name)
#     label_name = ast.literal_eval(label_name)
#     print(label_name)
#     device = torch.device("cuda:0")
#     image_list = os.listdir(image_dir)
#     random.shuffle(image_list)
#     for image_item in image_list:
#         # while True:
#         start_time = time.time()
#         path = os.path.join(image_dir, image_item)
#         im0 = cv2.imread(path)  # BGR
#         # im0原图
#         im, org_data = data_process_cv2(im0, imgsz)
#         # im:[1,3,640,640]变换维度后  org_data:[640,640,3]变换维度前
#         start_time0 = time.time()
#         y = session.run(output_names, {session.get_inputs()[0].name: im})
#         print("inference time : fo} ms", format((time.time() - start_time0) * 1000))
#         pred = torch.from_numpy(y[0]).to(device)
#         pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
#         print("spend time: fo} ms", format((time.time() - start_time) * 1000))
#         # res_img = post_process_yolov5(pred[0], org_data,label_name)
#
#         res_img = post_process_yolov5(pred[0], im0, label_name, org_data)
#         cv2.imshow("res", res_img)
#         cv2.waitKey(1)

from flask import Flask,request,render_template
import os
import cv2 as cv
#网页api
app = Flask(__name__, template_folder='.')
# 提供一个简单的用户界面，允许用户上传图片。
@app.route('/')
def home():
    return render_template('./templates/home.html')

@app.route('/predictio',methods=['POST','GET'])
def pre():
    if request.method=='POST':
        try:
            file=request.files.get('file')
            file_name=file.filename
            root='./static/img'
            path=f'{root}/{file_name}'
            file.save(path)
            # img=read_img(path)

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


            im0 = cv2.imread(path)  # BGR
            # im0原图
            im, org_data = data_process_cv2(im0, imgsz)
            # im:[1,3,640,640]变换维度后  org_data:[640,640,3]变换维度前
            y = session.run(output_names, {session.get_inputs()[0].name: im})
            pred = torch.from_numpy(y[0]).to(device)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
            # res_img = post_process_yolov5(pred[0], org_data,label_name)

            res_img = post_process_yolov5(pred[0], im0, label_name, org_data)
            # cv2.imshow("res", res_img)
            cv2.imwrite('static/img/cs.jpg',res_img)
            pre = os.path.split(path)[1]

            path = 'static/img/cs.jpg'
            # cv2.waitKey(0)

        except Exception as f:
            return f'{f}'

    return render_template('./templates/Prediction.html',user_image=f'{path}',product=f'{pre}')


# 设置上传文件夹
app.config['UPLOAD_FOLDER']='./static/img'
# 接收用户上传的图片文件。
app.config['ALLOWED_EXTENSIONS']={'png','jpg','jpeg'} # 设置允许的文件扩展名
app.config['MAX_CONTENT_LENGTH']= 5 * 1024 * 1024
app.run(host='0.0.0.0',port=6008,debug=True)
#

