# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('../')
import common
import time
import pycuda.autoinit
import tensorrt as trt
import torch
from dataloader import DataLoader
import cv2
from tool import non_max_suppression

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class trt_engine:
    def __init__(self,engine_path="yolov5.engine"):
        f = open(engine_path,'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

        # context = engine.create_execution_context()
        print(engine.get_binding_shape(0))
        print(engine.get_binding_shape(1))


        self.context = engine.create_execution_context()

        origin_inputshape = self.context.get_binding_shape(0)
        self.context.set_binding_shape(0,(origin_inputshape))
        self.inputs,self.outputs,self.bindings,self.stream = common.allocate_buffers(engine,self.context)
        self.__dict__.update(locals())

    def trt_inference(self,input_image):
        self.cuda_ctx = pycuda.autoinit.context
        self.cuda_ctx.push()
        self.inputs[0].host = input_image
        start = time.time()
        trt_outputs = common.do_inference(self.context,bindings=self.bindings,
                                          inputs=self.inputs,outputs=self.outputs,
                                          stream=self.stream,batch_size=1)
        end = time.time()

        if self.cuda_ctx:
            self.cuda_ctx.pop()

        prediction = trt_outputs[0].reshape(1,25200,85)
        prediction = torch.tensor(prediction)
        result = non_max_suppression(prediction,conf_thres=0.25,iou_thres=0.45,classes=None,
                                     agnostic=False,multi_label=False,labels=(),max_det=300,nm=0)
        return result

if __name__ == '__main__':

    trt = trt_engine()
    dataloader = DataLoader(image_size=[640,640])

    image_dir = r'C:\Users\26296\Desktop\v5\datasets\coco128\images\train2017'
    image_list = os.listdir(image_dir)
    for image_item in image_list:
        start_time = time.time()
        image_path = os.path.join(image_dir,image_item)
        src = cv2.imread(image_path)
        input_image,org_image = dataloader.data_process_cv2(src)
        predict = trt.trt_inference(input_image)
        print('spend time:{0} ms'.format((time.time() - start_time) * 1000))
        res_img = dataloader.post_process_yolov5(predict[0],org_image)
        cv2.imshow('res',res_img)
        cv2.waitKey()
