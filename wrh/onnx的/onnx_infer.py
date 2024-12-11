import ast
import time
import cv2
import onnx
import onnxruntime as rt
import torch
import os
from tool import *

def onnx_load(model):
    session = rt.InferenceSession(model,providers='cpu')
    input_names = session.get_inputs()[0].name
    output_names = session.get_outputs()[0].name
    return session,input_names,output_names

if __name__ == "__main__":
    model = 'yolov5s.onnx'
    img_dir = './images'
    img_size = [640,640]
    session,input_names,output_names = onnx_load(model)
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        start_time = time.time()
        path = os.path.join(img_dir,img_name)
        ori_img = cv2.imread(path)
        img,img0 = data_process_cv2(ori_img,img_size)
        res = session.run([output_names],{input_names:img})
        pred = torch.from_numpy(res[0]).cpu()
        pred = non_max_suppression(pred,conf_thres=0.5,iou_thres=0.5,max_det=1000)
        print("spend time:{0} ms".format(time.time() - start_time))
        label_names = session.get_modelmeta().custom_metadata_map['names']
        label_names = eval(label_names)
        res_img = post_process_yolov5(pred[0], ori_img ,img.shape[2:],label_names)
        cv2.imshow('res',res_img)
        cv2.waitKey(10)