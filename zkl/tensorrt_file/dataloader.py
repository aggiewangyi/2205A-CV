# -*- coding: utf-8 -*-
import numpy as np
import yaml
from PIL import Image
import copy
import cv2
from tool import scale_boxes
from tool import yaml_load
import math

def resize_image(image,size):
    iw,ih = image.size
    w,h = size

    scale = min(w/iw,h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh))
    new_image = Image.new('RGB',size,(128,128,128))
    new_image.paste(image,((w-nw)//2,(h-nh)//2))

    return new_image,nw,nh

def resize_image_cv2(image,size):
    ih,iw,ic = image.shape
    w,h = size

    scale = min(w/iw,h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image,(nw,nh))
    new_image = np.ones((size[0],size[1],3),dtype='uint8')*128
    start_h = (h-nh)/2
    start_w = (w-nw)/2

    end_h = size[1] - start_h
    end_w = size[0] - start_w

    new_image[int(start_h):int(end_h),int(start_w):int(end_w)] = image

    return new_image,nw,nh

def preprocess_input(image):
    image /= 255.0
    return image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def box_label(im, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = 2
    p1,p2 = (int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
    cv2.rectangle(im,p1,p2,color,thickness=lw,lineType=cv2.LINE_AA)
    if label:
        tf = max(lw-1,1)
        w,h = cv2.getTextSize(label,0,fontScale=lw/3,thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] +  w,p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im,p1,p2,color,-1,cv2.LINE_AA)
        cv2.putText(im,label,(p1[0],p1[1]-2 if outside else p1[1] + h + 2),0,lw/3,txt_color,
                    thickness=tf,lineType=cv2.LINE_AA)
    return im

class Colors:
    def __init__(self) -> None:
        hexs = ("FF3838","FF9D97","FF701F","FFB21D","CFD231","48F90A","92CC17",
                "3DDB86","1A9334","00D4BB","2C99A8","00C2FF","344593","6473FF",
            "0018EC","8438FF","520085","CB38FF","FF95C8","FF37C7")
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i)  % self.n]
        return (c[2],c[1],c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2],16) for i in (0,2,4))

class DataLoader():
    def __init__(self,image_size=[473,473]):
        self.colors = [(0,0,0),(128,0,0),(0,128,0),(128,128,0),
                       (0,0,128),(128,0,128),(0,128,128),(128,128,128),
                       (64,0,0),(192,0,0),(64,128,0),(192,128,0),(64,0,128),(192,0,128),
                       (64,128,128),(192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),
                       (0,64,128),(128,64,12),]
        self.input_shape = image_size

    def data_process(self,image):
        image = cvtColor(image)#确保图片是RGB格式

        self.origginal_h = np.array(image).shape[0]
        self.origginal_w = np.array(image).shape[1]

        '对图片进行缩放和填充'
        image_data,self.nw,self.nh = resize_image(image,(self.input_shape[1],self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,np.float32)),(2,0,1)),0)
        image_data = np.ascontiguousarray(image_data)

        return image_data


    def data_process_cv2(self,frame):
        self.origginal_h = frame.shape[0]
        self.origginal_w = frame.shape[1]

        image_data,self.nw,self.nh = resize_image_cv2(frame,(self.input_shape[1],self.input_shape[0]))
        org_data = image_data.copy()

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,np.float32)),(2,0,1)),0)
        image_data = np.ascontiguousarray(image_data)

        return image_data,org_data

    def post_process(self,pr):

        nh = self.nh
        nw =  self.nw

        input_shape = self.input_shape
        original_h = self.origginal_h
        original_w = self.origginal_w

        '将灰色部分截取掉'
        pr = pr[int((input_shape[0]-nh)//2):int((input_shape[0]-nh)//2+nh),
             int((input_shape[1]-nw)//2):int((input_shape[1]-nw)//2+nw)]

        '进行图片的resize'
        pr = cv2.resize(pr,(original_w,original_h),interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=1)

        pr_1d = pr.reshape(-1)
        counts = np.bincount(pr_1d)

        label_index = np.argsort(-counts)

        seg_img = np.reshape(np.array(self.colors,np.uint8)[np.reshape(pr,[-1])],[original_h,original_w,-1])

        return seg_img,label_index

    def post_process_yolov5(self,det,im,label_path="coco_label.yaml"):
        if len(det):
            det[:,:4] = scale_boxes(im.shape[:2],det[:,:4],im.shape).round()
            names = yaml_load(label_path)['names']
            colors = Colors()
            for *xyxy,conf,cls in reversed(det):
                c = int(cls)
                label = names[c]
                box_label(im,xyxy,label,color=colors(c,True))
        return im

    def save_data(self,pred):
        pred_np = pred.cpu().numpy()
        with open('./img_out/trt_output.txt','w') as outfile:
            np.savetxt(outfile,pred_np[0][8],fmt='%f',delimiter=" ")


