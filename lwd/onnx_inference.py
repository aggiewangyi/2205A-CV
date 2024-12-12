import os
import random
import cv2
import onnxruntime
import torch.cuda
from tool import *
import time

def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w,providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    return session,output_names
if __name__ == '__main__':
    w = 'yolov5s.onnx'
    image_dir = r'E:\study6\study\coco128\images\train2017'
    imgsz = [640,640]
    session,output_names = onnx_load(w)
    device = torch.device('cuda:0')
    image_list = os.listdir(image_dir)
    random.shuffle(image_list)
    for image_item in image_list:
        start_time = time.time()
        path = os.path.join(image_dir,image_item)
        im0 = cv2.imread(path) # BGR
        im,org_data = data_process_cv2(im0,imgsz)
        y = session.run(output_names,{session.get_inputs()[0].name:im})
        pred = torch.from_numpy(y[0]).to(device)
        pred = non_max_suppression(pred,conf_thres=0.25,iou_thres=0.45,max_det=1000)
        print("spend time:{0} ms".format((time.time()-start_time)*1000))
        res_img = post_process_yolov5(pred[0],org_data,label_path='coco128.yaml')
        cv2.imshow('res',res_img)
        cv2.waitKey(0)